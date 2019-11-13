#include <obs-module.h>
#include <util/circlebuf.h>
#include <Neuro.h>

#ifndef SEC_TO_NSEC
#define SEC_TO_NSEC 1000000000ULL
#endif

#ifndef MSEC_TO_NSEC
#define MSEC_TO_NSEC 1000000ULL
#endif

#define SETTING_ALPHA "alpha"
#define SETTING_ALPHA_TEXT "Style ratio"
#define SETTING_DELAY_NAME "delay_ms"
#define SETTING_DELAY_TEXT "Delay Ms"

using namespace Neuro;

static ModelBase* generator = nullptr;
static Placeholder* input = nullptr;
static TensorLike* stylizedContentPre = nullptr;

class OutputScale : public LayerBase
{
public:
    OutputScale(const string& name = "") : LayerBase(__FUNCTION__, Shape(), name) {}
protected:
    virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override { return { multiply(inputNodes[0], 150.f) }; }
};

static ModelBase* create_generator_model(TensorLike* input)
{
    NameScope scope("generator");

    auto training = new Constant(0);

    auto residual_block = [&](TensorLike* x, int num)
    {
        auto shortcut = x;
        x = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "resi_conv_" + to_string(num) + "_1"))->Call(x)[0];
        x = (new InstanceNormalization("resi_normal_" + to_string(num) + "_1"))->Call(x, training)[0];
        x = (new Activation(new ReLU(), "resi_relu_" + to_string(num) + "_1"))->Call(x)[0];
        x = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "resi_conv_" + to_string(num) + "_2"))->Call(x)[0];
        x = (new InstanceNormalization("resi_normal_" + to_string(num) + "_2"))->Call(x, training)[0];
        auto m = (new Merge(SumMerge, nullptr, "resi_add_" + to_string(num)))->Call({ x, shortcut })[0];
        return m;
    };

    auto input_o = new Input(input, "input_o");

    auto c1 = (new Conv2D(32, 9, 1, Tensor::GetPadding(Same, 9), nullptr, NCHW, "conv_1"))->Call(input_o->Outputs())[0];
    c1 = (new InstanceNormalization("normal_1"))->Call(c1, training)[0];
    c1 = (new Activation(new ReLU(), "relu_1"))->Call(c1)[0];

    auto c2 = (new Conv2D(64, 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_2"))->Call(c1)[0];
    c2 = (new InstanceNormalization("normal_2"))->Call(c2, training)[0];
    c2 = (new Activation(new ReLU(), "relu_2"))->Call(c2)[0];

    auto c3 = (new Conv2D(128, 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_3"))->Call(c2)[0];
    c3 = (new InstanceNormalization("normal_3"))->Call(c3, training)[0];
    c3 = (new Activation(new ReLU(), "relu_3"))->Call(c3)[0];

    auto r1 = residual_block(c3, 1);
    auto r2 = residual_block(r1, 2);
    auto r3 = residual_block(r2, 3);
    auto r4 = residual_block(r3, 4);
    auto r5 = residual_block(r4, 5);

    auto d1 = (new UpSampling2D(2, "up_1"))->Call(r5)[0];
    d1 = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_4"))->Call(d1)[0];
    d1 = (new InstanceNormalization("normal_4"))->Call(d1, training)[0];
    d1 = (new Activation(new ReLU(), "relu_4"))->Call(d1)[0];

    auto d2 = (new UpSampling2D(2, "up_2"))->Call(d1)[0];
    d2 = (new Conv2D(32, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_5"))->Call(d2)[0];
    d2 = (new InstanceNormalization("normal_5"))->Call(d2, training)[0];
    d2 = (new Activation(new ReLU(), "relu_5"))->Call(d2)[0];

    auto c4 = (new Conv2D(3, 9, 1, Tensor::GetPadding(Same, 9), nullptr, NCHW, "conv_6"))->Call(d2)[0];
    c4 = (new InstanceNormalization("normal_6"))->Call(c4, training)[0];
    c4 = (new Activation(new Tanh(), "tanh_1"))->Call(c4)[0];
    c4 = (new OutputScale("output"))->Call(c4)[0];

    return new Flow(input_o->Outputs(), { c4 }, "generator_model");
}

static void RGB_2_I420(const uint8_t* rgb, obs_source_frame* frame)
{
    uint32_t image_size = frame->width * frame->height;
    uint32_t upos = 0;
    uint32_t vpos = 0;
    uint32_t i = 0;

    for (uint32_t y = 0; y < frame->height; ++y)
    {
        if (!(y & 1))
        {
            for (size_t x = 0; x < frame->width; x += 2)
            {
                uint8_t r = rgb[i], g = rgb[i + 1], b = rgb[i + 2];
                rgb += 3;

                frame->data[0][i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                frame->data[1][upos++] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
                frame->data[2][vpos++] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;

                r = rgb[i];
                g = rgb[i + 1];
                b = rgb[i + 2];
                rgb += 3;

                frame->data[0][i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
            }
        }
        else
        {
            for (size_t x = 0; x < frame->width; x += 1)
            {
                uint8_t r = rgb[i], g = rgb[i + 1], b = rgb[i + 2];
                rgb += 3;

                frame->data[0][i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
            }
        }
    }
}

static void RGB_2_I420(const Tensor& t, obs_source_frame* frame)
{
    uint32_t image_size = frame->width * frame->height;
    uint32_t upos = 0;
    uint32_t vpos = 0;
    uint32_t i = 0;

    for (uint32_t y = 0; y < frame->height; ++y)
    {
        if (!(y & 1))
        {
            for (uint32_t x = 0; x < frame->width; x += 2)
            {
                uint8_t r = (uint8_t)t(x, y, 0), g = (uint8_t)t(x, y, 1), b = (uint8_t)t(x, y, 2);
                
                frame->data[0][i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                frame->data[1][upos++] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
                frame->data[2][vpos++] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;

                r = (uint8_t)t(x, y, 0);
                g = (uint8_t)t(x, y, 1);
                b = (uint8_t)t(x, y, 2);

                frame->data[0][i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
            }
        }
        else
        {
            for (uint32_t x = 0; x < frame->width; x += 1)
            {
                uint8_t r = (uint8_t)t(x, y, 0), g = (uint8_t)t(x, y, 1), b = (uint8_t)t(x, y, 2);

                frame->data[0][i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
            }
        }
    }
}

#define SCALEYUV(v) (((v)+128000)/256000)

static int rcoeff(int y, int u, int v) { return 298082 * y + 0 * u + 408583 * v; }
static int gcoeff(int y, int u, int v) { return 298082 * y - 100291 * u - 208120 * v; }
static int bcoeff(int y, int u, int v) { return 298082 * y + 516411 * u + 0 * v; }

int clamp(int vv)
{
    if (vv < 0)
        return 0;
    if (vv > 255)
        return 255;
    return vv;
}

static void I420_2_RGB(const obs_source_frame* frame, uint8_t* rgb)
{
    uint8_t* py = frame->data[0];
    uint8_t* pu = frame->data[1];
    uint8_t* pv = frame->data[2];

    for (uint32_t j = 0; j < frame->height; ++j)
    {
        for (uint32_t i = 0; i < frame->width; ++i)
        {
            int y = py[i] - 16;
            int u = pu[i / 2] - 128;
            int v = pv[i / 2] - 128;
            rgb[0] = clamp(SCALEYUV(rcoeff(y, u, v)));
            rgb[1] = clamp(SCALEYUV(gcoeff(y, u, v)));
            rgb[2] = clamp(SCALEYUV(bcoeff(y, u, v)));
            rgb += 3;
        }

        py += frame->linesize[0];
        if (j & 1)
        {
            pu += frame->linesize[1];
            pv += frame->linesize[2];
        }
    }
}

static void I420_2_RGB(const obs_source_frame* frame, Tensor& t)
{
    uint8_t* py = frame->data[0];
    uint8_t* pu = frame->data[1];
    uint8_t* pv = frame->data[2];

    for (uint32_t j = 0; j < frame->height; ++j)
    {
        for (uint32_t i = 0; i < frame->width; ++i)
        {
            int y = py[i] - 16;
            int u = pu[i / 2] - 128;
            int v = pv[i / 2] - 128;
            t(i, j, 0) = (float)clamp(SCALEYUV(rcoeff(y, u, v)));
            t(i, j, 1) = (float)clamp(SCALEYUV(gcoeff(y, u, v)));
            t(i, j, 2) = (float)clamp(SCALEYUV(bcoeff(y, u, v)));
        }

        py += frame->linesize[0];
        if (j & 1)
        {
            pu += frame->linesize[1];
            pv += frame->linesize[2];
        }
    }
}

struct style_data
{
    obs_source_t* context;

    /* contains struct obs_source_frame* */
    circlebuf video_frames;

    /* stores the audio data */
    circlebuf audio_frames;
    obs_audio_data audio_output;

    uint64_t last_video_ts;
    uint64_t last_audio_ts;
    uint64_t interval;
    uint64_t samplerate;
    bool video_delay_reached;
    bool audio_delay_reached;
    bool reset_video;
    bool reset_audio;
};

static const char* style_filter_name(void* unused)
{
    UNUSED_PARAMETER(unused);
    return "Style Filter";
}

static void free_video_data(style_data* filter, obs_source_t* parent)
{
    while (filter->video_frames.size)
    {
        obs_source_frame* frame;

        circlebuf_pop_front(&filter->video_frames, &frame, sizeof(obs_source_frame*));
        obs_source_release_frame(parent, frame);
    }
}

static inline void free_audio_packet(obs_audio_data* audio)
{
    for (size_t i = 0; i < MAX_AV_PLANES; i++)
        bfree(audio->data[i]);
    memset(audio, 0, sizeof(*audio));
}

static void free_audio_data(style_data* filter)
{
    while (filter->audio_frames.size) {
        obs_audio_data audio;

        circlebuf_pop_front(&filter->audio_frames, &audio,
            sizeof(obs_audio_data));
        free_audio_packet(&audio);
    }
}

static void style_filter_update(void* data, obs_data_t* settings)
{
    style_data* filter = (style_data*)data;
    uint64_t new_interval =
        (uint64_t)obs_data_get_int(settings, SETTING_DELAY_NAME)* 
        MSEC_TO_NSEC;

    if (new_interval < filter->interval)
        free_video_data(filter, obs_filter_get_parent(filter->context));

    filter->reset_audio = true;
    filter->reset_video = true;
    filter->interval = new_interval;
    filter->video_delay_reached = false;
    filter->audio_delay_reached = false;
}

static void* style_filter_create(obs_data_t* settings, obs_source_t* context)
{
    style_data* filter = (style_data*)bzalloc(sizeof(style_data));
    obs_audio_info oai;

    filter->context = context;
    style_filter_update(filter, settings);

    obs_get_audio_info(&oai);
    filter->samplerate = oai.samples_per_sec;

    Tensor::SetForcedOpMode(GPU);

    return filter;
}

static void style_filter_destroy(void* data)
{
    style_data* filter = (style_data*)data;

    free_audio_packet(&filter->audio_output);
    circlebuf_free(&filter->video_frames);
    circlebuf_free(&filter->audio_frames);
    bfree(data);
}

static obs_properties_t* style_filter_properties(void* data)
{
    obs_properties_t* props = obs_properties_create();
    
    obs_properties_add_float(props, SETTING_ALPHA, SETTING_ALPHA_TEXT, 0, 1, 0.01);
    obs_property_t* p = obs_properties_add_int(props, SETTING_DELAY_NAME, SETTING_DELAY_TEXT, 0, 20000, 1);
    obs_property_int_set_suffix(p, " ms");

    UNUSED_PARAMETER(data);
    return props;
}

static void style_filter_remove(void* data, obs_source_t* parent)
{
    style_data* filter = (style_data*)data;

    free_video_data(filter, parent);
    free_audio_data(filter);
}

/* due to the fact that we need timing information to be consistent in order to
* measure the current interval of data, if there is an unexpected hiccup or
* jump with the timestamps, reset the cached delay data and start again to
* ensure that the timing is consistent */
static inline bool is_timestamp_jump(uint64_t ts, uint64_t prev_ts)
{
    return ts < prev_ts || (ts - prev_ts) > SEC_TO_NSEC;
}

static obs_source_frame* style_filter_video(void* data, obs_source_frame* frame)
{
    style_data* filter = (style_data*)data;
    obs_source_t* parent = obs_filter_get_parent(filter->context);
    obs_source_frame* output;
    uint64_t cur_interval;

    if (filter->reset_video ||
        is_timestamp_jump(frame->timestamp, filter->last_video_ts)) {
        free_video_data(filter, parent);
        filter->video_delay_reached = false;
        filter->reset_video = false;
    }

    filter->last_video_ts = frame->timestamp;

    static bool capture_frame = false;

    if (capture_frame)
    {
        if (!generator)
        {
            input = new Placeholder(Shape(frame->width, frame->height, 3), "input");
            generator = create_generator_model(VGG16::Preprocess(input, NCHW));
            stylizedContentPre = generator->Outputs()[0];
            generator->LoadWeights("e:/mosaic_weights.h5", false, true);
        }

        if (frame->format == VIDEO_FORMAT_I420)
        {
            /*vector<uint8_t> rgb_data;
            rgb_data.resize(frame->width * frame->height * 3);*/
            Tensor frameData(Shape(frame->width, frame->height, 3));
            frameData.OverrideHost();

            I420_2_RGB(frame, frameData);

            auto results = Session::Default()->Run({ stylizedContentPre }, { { input, &frameData } });
            auto frameDataStylized = *results[0];
            VGG16::DeprocessImage(frameDataStylized, NCHW);

            frameDataStylized.SaveAsImage("e:/_frame.jpg", false);

            RGB_2_I420(frameDataStylized, frame);

            /*Tensor frameT = LoadImage(&rgb_data[0], frame->width, frame->height, EPixelFormat::RGB);
            frameT.SaveAsImage("e:/_frame.jpg", false);

            static Tensor noise = Uniform::Random(-50, 50, frameT.GetShape());

            frameT.Add(noise);*/

            /*for (size_t i = 0; i < rgb_data.size(); ++i)
                rgb_data[i] = clamp(rgb_data[i] + GlobalRng().Next(-50, 50));

            RGB_2_I420(&rgb_data[0], frame);*/
        }        
    }

    circlebuf_push_back(&filter->video_frames, &frame, sizeof(obs_source_frame*));
    circlebuf_peek_front(&filter->video_frames, &output, sizeof(obs_source_frame*));

    cur_interval = frame->timestamp - output->timestamp;
    if (!filter->video_delay_reached && cur_interval < filter->interval)
        return NULL;

    circlebuf_pop_front(&filter->video_frames, NULL, sizeof(obs_source_frame*));

    if (!filter->video_delay_reached)
        filter->video_delay_reached = true;

    return output;
}

/* NOTE: Delaying audio shouldn't be necessary because the audio subsystem will
* automatically sync audio to video frames */

/* #define DELAY_AUDIO */

#ifdef DELAY_AUDIO
static obs_audio_data* 
style_filter_audio(void* data, obs_audio_data* audio)
{
    style_data* filter = data;
    obs_audio_data cached =* audio;
    uint64_t cur_interval;
    uint64_t duration;
    uint64_t end_ts;

    if (filter->reset_audio ||
        is_timestamp_jump(audio->timestamp, filter->last_audio_ts)) {
        free_audio_data(filter);
        filter->audio_delay_reached = false;
        filter->reset_audio = false;
    }

    filter->last_audio_ts = audio->timestamp;

    duration = (uint64_t)audio->frames * SEC_TO_NSEC / filter->samplerate;
    end_ts = audio->timestamp + duration;

    for (size_t i = 0; i < MAX_AV_PLANES; i++) {
        if (!audio->data[i])
            break;

        cached.data[i] =
            bmemdup(audio->data[i], audio->frames * sizeof(float));
    }

    free_audio_packet(&filter->audio_output);

    circlebuf_push_back(&filter->audio_frames, &cached, sizeof(cached));
    circlebuf_peek_front(&filter->audio_frames, &cached, sizeof(cached));

    cur_interval = end_ts - cached.timestamp;
    if (!filter->audio_delay_reached && cur_interval < filter->interval)
        return NULL;

    circlebuf_pop_front(&filter->audio_frames, NULL, sizeof(cached));
    memcpy(&filter->audio_output, &cached, sizeof(cached));

    if (!filter->audio_delay_reached)
        filter->audio_delay_reached = true;

    return &filter->audio_output;
}
#endif

obs_source_info style_filter = (style_filter = obs_source_info(),
    style_filter.id = "style_filter",
    style_filter.type = OBS_SOURCE_TYPE_FILTER,
    style_filter.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
    style_filter.get_name = style_filter_name,
    style_filter.create = style_filter_create,
    style_filter.destroy = style_filter_destroy,
    style_filter.update = style_filter_update,
    style_filter.get_properties = style_filter_properties,
    style_filter.filter_video = style_filter_video,
#ifdef DELAY_AUDIO
    style_filter.filter_audio = style_filter_audio,
#endif
    style_filter.filter_remove = style_filter_remove,
    style_filter
);

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-stylefilter", "en-US")

bool obs_module_load(void)
{
    obs_register_source(&style_filter);

    return true;
}

void obs_module_unload(void)
{
}
