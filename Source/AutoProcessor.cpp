#include "AutoProcessor.h"

#ifndef AUTO_IS_MASTER
 #define AUTO_IS_MASTER 1
#endif

//==============================================================================
void Biquad::setPeak (double fs, double f, double q, double gainDb)
{
    const double A = std::pow (10.0, gainDb / 40.0);
    const double w0 = juce::MathConstants<double>::twoPi * f / fs;
    const double alpha = std::sin (w0) / (2.0 * q);
    const double c = std::cos (w0);
    set (1 + alpha * A, -2 * c, 1 - alpha * A, 1 + alpha / A, -2 * c, 1 - alpha / A);
}

void Biquad::setHighPass (double fs, double f, double q)
{
    const double w0 = juce::MathConstants<double>::twoPi * f / fs;
    const double alpha = std::sin (w0) / (2.0 * q);
    const double c = std::cos (w0);
    set ((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + alpha, -2 * c, 1 - alpha);
}

void Biquad::setHighShelf (double fs, double f, double gainDb)
{
    const double A = std::pow (10.0, gainDb / 40.0);
    const double w0 = juce::MathConstants<double>::twoPi * f / fs;
    const double c = std::cos (w0);
    const double alpha = std::sin (w0) / 2.0 * std::sqrt (2.0);
    const double t = 2.0 * std::sqrt (A) * alpha;
    set (A * ((A + 1) + (A - 1) * c + t),
         -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - t),
         (A + 1) - (A - 1) * c + t,
         2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - t);
}

//==============================================================================
static const float bandFreqs[10] = { 31.f, 62.f, 125.f, 250.f, 500.f, 1000.f, 2000.f, 4000.f, 8000.f, 16000.f };

juce::AudioProcessorValueTreeState::ParameterLayout AutoProcessor::createLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout l;
    l.add (std::make_unique<AudioParameterFloat> (ParameterID { "amount", 1 }, "Amount",
           NormalisableRange<float> (0.f, 100.f, 0.1f), 70.f, AudioParameterFloatAttributes().withLabel ("%")));
    l.add (std::make_unique<AudioParameterFloat> (ParameterID { "target", 1 }, "Target Loudness",
           NormalisableRange<float> (-24.f, -6.f, 0.1f), AUTO_IS_MASTER ? -14.f : -18.f,
           AudioParameterFloatAttributes().withLabel ("LUFS")));
    l.add (std::make_unique<AudioParameterFloat> (ParameterID { "ceiling", 1 }, "Ceiling",
           NormalisableRange<float> (-6.f, 0.f, 0.1f), -1.f, AudioParameterFloatAttributes().withLabel ("dB")));
    l.add (std::make_unique<AudioParameterFloat> (ParameterID { "width", 1 }, "Stereo Width",
           NormalisableRange<float> (0.f, 200.f, 1.f), 100.f, AudioParameterFloatAttributes().withLabel ("%")));
    l.add (std::make_unique<AudioParameterBool> (ParameterID { "bypass", 1 }, "Bypass", false));
    return l;
}

AutoProcessor::AutoProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "STATE", createLayout())
{
    pAmount  = apvts.getRawParameterValue ("amount");
    pTarget  = apvts.getRawParameterValue ("target");
    pCeiling = apvts.getRawParameterValue ("ceiling");
    pWidth   = apvts.getRawParameterValue ("width");
    pBypass  = apvts.getRawParameterValue ("bypass");
}

bool AutoProcessor::isBusesLayoutSupported (const BusesLayout& l) const
{
    return l.getMainInputChannelSet() == juce::AudioChannelSet::stereo()
        && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void AutoProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    sr = sampleRate;
    juce::dsp::ProcessSpec spec { sampleRate, (juce::uint32) juce::jmax (1, samplesPerBlock), 2 };
    comp.prepare (spec);
    limiter.prepare (spec);
    comp.reset();
    limiter.reset();

    for (int c = 0; c < 2; ++c)
    {
        hp[c].setHighPass (sr, 25.0, 0.707);   hp[c].reset();
        kHp[c].setHighPass (sr, 38.0, 0.5);    kHp[c].reset();
        kShelf[c].setHighShelf (sr, 1500.0, 4.0); kShelf[c].reset();
        for (auto& b : eq[c]) { b = Biquad(); }
    }
    fifo.fill (0.f);
    fifoPos = 0;
    bandPow.fill (0.0);
    corrTarget.fill (0.f); corrSmooth.fill (0.f); corrApplied.fill (0.f);
    haveSpectrum = false;
    rmsAvg = 0.0; msAvg = 0.0; msInit = false;
    gainDb = 0.f; lastGain = 1.f;
}

//==============================================================================
void AutoProcessor::analyse()
{
    std::copy (fifo.begin(), fifo.end(), fftData.begin());
    std::fill (fftData.begin() + fftSize, fftData.end(), 0.f);
    window.multiplyWithWindowingTable (fftData.data(), (size_t) fftSize);
    fft.performFrequencyOnlyForwardTransform (fftData.data());

    double total = 0.0;
    double p[nBands];
    for (int b = 0; b < nBands; ++b)
    {
        const double lo = bandFreqs[b] / 1.41421356;
        const double hi = juce::jmin (bandFreqs[b] * 1.41421356, sr * 0.5);
        int i0 = juce::jlimit (1, fftSize / 2 - 1, (int) std::floor (lo * fftSize / sr));
        int i1 = juce::jlimit (i0 + 1, fftSize / 2, (int) std::ceil (hi * fftSize / sr));
        double s = 0.0;
        for (int i = i0; i < i1; ++i) s += (double) fftData[(size_t) i] * fftData[(size_t) i];
        p[b] = s / (double) (i1 - i0);
        total += p[b];
    }
    if (total < 1e-3) return; // silence: keep previous analysis

    for (int b = 0; b < nBands; ++b)
        bandPow[(size_t) b] = haveSpectrum ? 0.85 * bandPow[(size_t) b] + 0.15 * p[b] : p[b];
    haveSpectrum = true;
}

void AutoProcessor::updateEq (float amount, double blockSec)
{
    if (haveSpectrum)
    {
        double db[nBands], diff[nBands];
        double mean = 0.0;
        for (int b = 0; b < nBands; ++b)
        {
            db[b] = 10.0 * std::log10 (bandPow[(size_t) b] + 1e-12);
            diff[b] = db[b] - (-4.5 * b);           // target: pink-ish -4.5 dB/oct tilt
            if (b >= 1 && b <= 8) mean += diff[b];
        }
        mean /= 8.0;
        const float maxC = AUTO_IS_MASTER ? 3.0f : 5.0f;
        corrTarget[0] = 0.f;
        for (int b = 1; b < nBands; ++b)
            corrTarget[(size_t) b] = juce::jlimit (-maxC, maxC, (float) (-(diff[b] - mean))) * amount;
    }

    const float k = (float) (1.0 - std::exp (-blockSec / 1.5));
    for (int b = 1; b < nBands; ++b)
    {
        corrSmooth[(size_t) b] += (corrTarget[(size_t) b] - corrSmooth[(size_t) b]) * k;
        if (std::abs (corrSmooth[(size_t) b] - corrApplied[(size_t) b]) > 0.01f
            && bandFreqs[b] < sr * 0.45)
        {
            corrApplied[(size_t) b] = corrSmooth[(size_t) b];
            eq[0][b].setPeak (sr, bandFreqs[b], 1.0, corrApplied[(size_t) b]);
            eq[1][b].copyCoeffs (eq[0][b]);
        }
    }
}

//==============================================================================
void AutoProcessor::processBlock (juce::AudioBuffer<float>& buf, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int n = buf.getNumSamples();
    if (n <= 0 || buf.getNumChannels() < 2 || pBypass->load() > 0.5f) return;

    const double blockSec = (double) n / sr;
    const float amount = pAmount->load() / 100.0f;
    float* L = buf.getWritePointer (0);
    float* R = buf.getWritePointer (1);

    // 1) spectrum analysis of the input (mono sum)
    for (int i = 0; i < n; ++i)
    {
        fifo[(size_t) fifoPos++] = 0.5f * (L[i] + R[i]);
        if (fifoPos >= fftSize) { analyse(); fifoPos = 0; }
    }

    // 2) auto EQ (+ rumble filter)
    updateEq (amount, blockSec);
    for (int c = 0; c < 2; ++c)
    {
        float* d = buf.getWritePointer (c);
        for (int i = 0; i < n; ++i)
        {
            float x = hp[c].process (d[i]);
            for (int b = 1; b < nBands; ++b) x = eq[c][b].process (x);
            d[i] = x;
        }
    }

    // 3) auto compressor (threshold follows the program level)
    {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += 0.5 * ((double) L[i] * L[i] + (double) R[i] * R[i]);
        const double ms = sum / n;
        const double a = std::exp (-blockSec / 1.0);
        rmsAvg = a * rmsAvg + (1.0 - a) * ms;
        const float rmsDb = (float) (10.0 * std::log10 (rmsAvg + 1e-12));
        const float baseRatio = AUTO_IS_MASTER ? 2.0f : 3.0f;
        comp.setThreshold (juce::jlimit (-60.f, 0.f, rmsDb + (AUTO_IS_MASTER ? 0.f : -2.f)));
        comp.setRatio (1.0f + (baseRatio - 1.0f) * amount);
        comp.setAttack (AUTO_IS_MASTER ? 30.f : 15.f);
        comp.setRelease (AUTO_IS_MASTER ? 200.f : 120.f);
        juce::dsp::AudioBlock<float> block (buf);
        juce::dsp::ProcessContextReplacing<float> ctx (block);
        comp.process (ctx);
    }

    // 4) loudness-targeting gain (slow closed loop, K-weighted measure)
    const float newGain = juce::Decibels::decibelsToGain (gainDb);
    buf.applyGainRamp (0, n, lastGain, newGain);
    lastGain = newGain;
    {
        double sumSq = 0.0;
        for (int c = 0; c < 2; ++c)
        {
            const float* d = buf.getReadPointer (c);
            double s = 0.0;
            for (int i = 0; i < n; ++i)
            {
                const float y = kShelf[c].process (kHp[c].process (d[i]));
                s += (double) y * y;
            }
            sumSq += s / n;
        }
        if (! msInit) { msAvg = sumSq; msInit = true; }
        const double a = std::exp (-blockSec / 3.0);
        msAvg = a * msAvg + (1.0 - a) * sumSq;
        const double lufs = -0.691 + 10.0 * std::log10 (msAvg + 1e-12);
        if (lufs > -55.0)
        {
            const double err = (double) pTarget->load() - lufs;
            gainDb = juce::jlimit (-18.f, 18.f, gainDb + (float) (err * blockSec * 0.15));
        }
    }

    // 5) stereo width
    const float w = pWidth->load() / 100.0f;
    if (std::abs (w - 1.0f) > 0.001f)
        for (int i = 0; i < n; ++i)
        {
            const float m = 0.5f * (L[i] + R[i]);
            const float s = 0.5f * (L[i] - R[i]) * w;
            L[i] = m + s; R[i] = m - s;
        }

    // 6) limiter + hard safety clamp at the ceiling
    const float ceilDb = pCeiling->load();
    limiter.setThreshold (ceilDb);
    limiter.setRelease (80.f);
    {
        juce::dsp::AudioBlock<float> block (buf);
        juce::dsp::ProcessContextReplacing<float> ctx (block);
        limiter.process (ctx);
    }
    const float lim = juce::Decibels::decibelsToGain (ceilDb);
    for (int c = 0; c < 2; ++c)
    {
        float* d = buf.getWritePointer (c);
        for (int i = 0; i < n; ++i) d[i] = juce::jlimit (-lim, lim, d[i]);
    }
}

//==============================================================================
juce::AudioProcessorEditor* AutoProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor (*this);
}

void AutoProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    if (auto xml = apvts.copyState().createXml()) copyXmlToBinary (*xml, dest);
}

void AutoProcessor::setStateInformation (const void* data, int size)
{
    if (auto xml = getXmlFromBinary (data, size))
        if (xml->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new AutoProcessor(); }
