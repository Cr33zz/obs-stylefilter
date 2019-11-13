#include <obs-module.h>
#include <util/circlebuf.h>
#include <Neuro.h>

#ifndef SEC_TO_NSEC
#define SEC_TO_NSEC 1000000000ULL
#endif

#ifndef MSEC_TO_NSEC
#define MSEC_TO_NSEC 1000000ULL
#endif

#define SCALEYUV(v) (((v)+128000)/256000)

static int rcoeff(int y, int u, int v) { return 298082 * y + 0 * u + 408583 * v; }
static int gcoeff(int y, int u, int v) { return 298082 * y - 100291 * u - 208120 * v; }
static int bcoeff(int y, int u, int v) { return 298082 * y + 516411 * u + 0 * v; }

int clamp(int vv)
{
    if (vv < 0)
        return 0;
    else if (vv > 255)
        return 255;
    return vv;
}

#define SETTING_ALPHA "alpha"
#define SETTING_ALPHA_TEXT "Style ratio"
#define SETTING_DELAY_NAME "delay_ms"
#define SETTING_DELAY_TEXT "Delay Ms"

using namespace Neuro;

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

static void* style_filter_create(obs_data_t* settings,
    obs_source_t* context)
{
    style_data* filter = (style_data*)bzalloc(sizeof(style_data));
    obs_audio_info oai;

    filter->context = context;
    style_filter_update(filter, settings);

    obs_get_audio_info(&oai);
    filter->samplerate = oai.samples_per_sec;

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
        if (frame->format == VIDEO_FORMAT_I420)
        {
            vector<uint8_t> rgb_data;
            rgb_data.reserve(frame->width * frame->height * 3);

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
                    rgb_data.push_back(clamp(SCALEYUV(rcoeff(y, u, v))));
                    rgb_data.push_back(clamp(SCALEYUV(gcoeff(y, u, v))));
                    rgb_data.push_back(clamp(SCALEYUV(bcoeff(y, u, v))));
                }

                py += frame->linesize[0];
                if (j & 1)
                {
                    pu += frame->linesize[1];
                    pv += frame->linesize[2];
                }
            }

            Tensor frameT = LoadImage(&rgb_data[0], frame->width, frame->height, EPixelFormat::RGB);
            frameT.SaveAsImage("e:/_frame.jpg", false);
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