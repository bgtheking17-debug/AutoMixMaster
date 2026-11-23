#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>

// Allocation-free biquad (safe for the audio thread)
struct Biquad
{
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float z1 = 0, z2 = 0;

    float process (float x) noexcept
    {
        const float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
    void reset() noexcept { z1 = z2 = 0; }
    void copyCoeffs (const Biquad& o) noexcept { b0 = o.b0; b1 = o.b1; b2 = o.b2; a1 = o.a1; a2 = o.a2; }

    void setPeak (double fs, double f, double q, double gainDb);
    void setHighPass (double fs, double f, double q);
    void setHighShelf (double fs, double f, double gainDb);

private:
    void set (double B0, double B1, double B2, double A0, double A1, double A2) noexcept
    {
        b0 = (float) (B0 / A0); b1 = (float) (B1 / A0); b2 = (float) (B2 / A0);
        a1 = (float) (A1 / A0); a2 = (float) (A2 / A0);
    }
};

class AutoProcessor : public juce::AudioProcessor
{
public:
    AutoProcessor();
    ~AutoProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using juce::AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

private:
    static constexpr int nBands = 10;
    static constexpr int fftOrder = 12;
    static constexpr int fftSize = 1 << fftOrder;

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void analyse();
    void updateEq (float amount, double blockSec);

    juce::AudioProcessorValueTreeState apvts;
    std::atomic<float>* pAmount = nullptr;
    std::atomic<float>* pTarget = nullptr;
    std::atomic<float>* pCeiling = nullptr;
    std::atomic<float>* pWidth = nullptr;
    std::atomic<float>* pBypass = nullptr;

    double sr = 44100.0;

    // spectrum analysis
    juce::dsp::FFT fft { fftOrder };
    juce::dsp::WindowingFunction<float> window { (size_t) fftSize, juce::dsp::WindowingFunction<float>::hann };
    std::array<float, fftSize> fifo {};
    std::array<float, fftSize * 2> fftData {};
    int fifoPos = 0;
    std::array<double, nBands> bandPow {};
    bool haveSpectrum = false;

    // processing
    std::array<float, nBands> corrTarget {}, corrSmooth {}, corrApplied {};
    Biquad eq[2][nBands];
    Biquad hp[2], kHp[2], kShelf[2];
    juce::dsp::Compressor<float> comp;
    juce::dsp::Limiter<float> limiter;

    double rmsAvg = 0.0;
    double msAvg = 0.0;
    bool msInit = false;
    float gainDb = 0.0f, lastGain = 1.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AutoProcessor)
};
