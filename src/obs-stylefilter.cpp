#include <obs-module.h>
#include <Neuro.h>

#define SETTING_ENABLED_TEXT "Style enabled"
#define SETTING_ENABLED_NAME "enabled"
#define SETTING_ALPHA_TEXT "Style ratio"
#define SETTING_ALPHA_NAME "alpha"
#define SETTING_STYLE_FILE_NAME "Style"
#define SETTING_STYLE_FILE_TEXT "style"

#define STYLE_IMAGE_WIDTH 512
#define STYLE_IMAGE_HEIGHT 512

using namespace Neuro;

static ModelBase* generator = nullptr;
static Placeholder* input_content = nullptr;
static Placeholder* input_style = nullptr;
static Placeholder* input_alpha = nullptr;
static TensorLike* stylized = nullptr;

class AdaIN : public LayerBase
{
public:
    AdaIN(TensorLike* alpha, const string& name = "") : LayerBase(__FUNCTION__, Shape(), name), m_Alpha(alpha) {}
protected:
    virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override
    {
        auto contentFeat = inputNodes[0];
        auto styleFeat = inputNodes[1];

        auto styleMean = mean(styleFeat, _01Axes);
        auto styleStd = std_deviation(styleFeat, styleMean, _01Axes);

        auto normContentFeat = instance_norm(contentFeat, styleStd, styleMean, 0.00001f, training);
        return { add(multiply(normContentFeat, m_Alpha), multiply(contentFeat, add(negative(m_Alpha), 1))) };
    }

    TensorLike* m_Alpha;
};

static ModelBase* create_generator_model(TensorLike* contentPre, TensorLike* stylePre, TensorLike* alpha, Flow& vggEncoder)
{
    NameScope scope("generator");

    auto training = new Constant(0);

    auto inputContent = new Input(contentPre, "input_content");
    auto inputStyle = new Input(stylePre, "input_style");

    // encoder

    auto contentFeat = vggEncoder(inputContent->Outputs(), nullptr, "content_features").back();
    auto styleFeat = vggEncoder(inputStyle->Outputs(), nullptr, "style_features").back();

    // adaptive instance normalization

    auto adaInFeat = (new AdaIN(alpha))->Call({ contentFeat, styleFeat }, training, "ada_in")[0];

    // decoder

    auto d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block4_conv1"))->Call(adaInFeat)[0];
    d = (new UpSampling2D(2, "decode_up_1"))->Call(d)[0];

    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv4"))->Call(d)[0];
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv3"))->Call(d)[0];
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv2"))->Call(d)[0];
    d = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv1"))->Call(d)[0];
    d = (new UpSampling2D(2, "decode_up_2"))->Call(d)[0];

    d = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block2_conv2"))->Call(d)[0];
    d = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block2_conv1"))->Call(d)[0];
    d = (new UpSampling2D(2, "decode_up_3"))->Call(d)[0];

    d = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block1_conv2"))->Call(d)[0];
    d = (new Conv2D(3, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "decode_block1_conv1"))->Call(d)[0];

    return new Flow({ inputContent->Outputs()[0], inputStyle->Outputs()[0] }, { d, adaInFeat }, "generator_model");
}

//class OutputScale : public LayerBase
//{
//public:
//    OutputScale(const string& name = "") : LayerBase(__FUNCTION__, Shape(), name) {}
//protected:
//    virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override { return { multiply(inputNodes[0], 150.f) }; }
//};
//
//static ModelBase* create_generator_model(TensorLike* input)
//{
//    NameScope scope("generator");
//
//    auto training = new Constant(0);
//
//    auto residual_block = [&](TensorLike* x, int num)
//    {
//        auto shortcut = x;
//        x = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "resi_conv_" + to_string(num) + "_1"))->Call(x)[0];
//        x = (new InstanceNormalization("resi_normal_" + to_string(num) + "_1"))->Call(x, training)[0];
//        x = (new Activation(new ReLU(), "resi_relu_" + to_string(num) + "_1"))->Call(x)[0];
//        x = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "resi_conv_" + to_string(num) + "_2"))->Call(x)[0];
//        x = (new InstanceNormalization("resi_normal_" + to_string(num) + "_2"))->Call(x, training)[0];
//        auto m = (new Merge(SumMerge, nullptr, "resi_add_" + to_string(num)))->Call({ x, shortcut })[0];
//        return m;
//    };
//
//    auto input_o = new Input(input, "input_o");
//
//    auto c1 = (new Conv2D(32, 9, 1, Tensor::GetPadding(Same, 9), nullptr, NCHW, "conv_1"))->Call(input_o->Outputs())[0];
//    c1 = (new InstanceNormalization("normal_1"))->Call(c1, training)[0];
//    c1 = (new Activation(new ReLU(), "relu_1"))->Call(c1)[0];
//
//    auto c2 = (new Conv2D(64, 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_2"))->Call(c1)[0];
//    c2 = (new InstanceNormalization("normal_2"))->Call(c2, training)[0];
//    c2 = (new Activation(new ReLU(), "relu_2"))->Call(c2)[0];
//
//    auto c3 = (new Conv2D(128, 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_3"))->Call(c2)[0];
//    c3 = (new InstanceNormalization("normal_3"))->Call(c3, training)[0];
//    c3 = (new Activation(new ReLU(), "relu_3"))->Call(c3)[0];
//
//    auto r1 = residual_block(c3, 1);
//    auto r2 = residual_block(r1, 2);
//    auto r3 = residual_block(r2, 3);
//    auto r4 = residual_block(r3, 4);
//    auto r5 = residual_block(r4, 5);
//
//    auto d1 = (new UpSampling2D(2, "up_1"))->Call(r5)[0];
//    d1 = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_4"))->Call(d1)[0];
//    d1 = (new InstanceNormalization("normal_4"))->Call(d1, training)[0];
//    d1 = (new Activation(new ReLU(), "relu_4"))->Call(d1)[0];
//
//    auto d2 = (new UpSampling2D(2, "up_2"))->Call(d1)[0];
//    d2 = (new Conv2D(32, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_5"))->Call(d2)[0];
//    d2 = (new InstanceNormalization("normal_5"))->Call(d2, training)[0];
//    d2 = (new Activation(new ReLU(), "relu_5"))->Call(d2)[0];
//
//    auto c4 = (new Conv2D(3, 9, 1, Tensor::GetPadding(Same, 9), nullptr, NCHW, "conv_6"))->Call(d2)[0];
//    c4 = (new InstanceNormalization("normal_6"))->Call(c4, training)[0];
//    c4 = (new Activation(new Tanh(), "tanh_1"))->Call(c4)[0];
//    c4 = (new OutputScale("output"))->Call(c4)[0];
//
//    return new Flow(input_o->Outputs(), { c4 }, "generator_model");
//}

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

static void RGB_2_YUY2(const Tensor& t, obs_source_frame* frame)
{
    int Y, U, V;
    int r, g, b;
    for (uint32_t h = 0; h < frame->height; ++h)
    {
        for (uint32_t w = 0; w < frame->width; w++)
        {

            r = (int)t(w, h, 0);
            g = (int)t(w, h, 1);
            b = (int)t(w, h, 2);

            Y = clamp((0.257*r) + (0.504*g) + (0.098*b) + 16);
            U = clamp(-(0.148*r) - (0.291*g) + (0.439*b) + 128);
            V = clamp((0.439*r) - (0.368*g) - (0.071*b) + 128);

            if ((w & 1) == 0)
            {
                frame->data[0][h * frame->linesize[0] + w * 2 + 0] = Y;
                frame->data[0][h * frame->linesize[0] + w * 2 + 1] = U;
                frame->data[0][h * frame->linesize[0] + w * 2 + 3] = V;
            }
            else
            {
                frame->data[0][h * frame->linesize[0] + w * 2 + 0] = Y;
                frame->data[0][h * frame->linesize[0] + w * 2 - 1] = U;
                frame->data[0][h * frame->linesize[0] + w * 2 + 1] = V;
            }
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

static void YUY2_2_RGB(const obs_source_frame* frame, Tensor& t)
{
    uint8_t* p = frame->data[0];

    for (uint32_t h = 0; h < frame->height; ++h)
    {
        uint32_t w = 0;
        for (uint32_t i = 0; i < frame->width / 2; ++i)
        {
            int y0 = p[0];
            int u0 = p[1];
            int y1 = p[2];
            int v0 = p[3];
            p += 4;
            int c = y0 - 16;
            int d = u0 - 128;
            int e = v0 - 128;
            t(w, h, 0) = (float)clamp((298 * c + 409 * e + 128) >> 8); // red
            t(w, h, 1) = (float)clamp((298 * c - 100 * d - 208 * e + 128) >> 8); // green
            t(w, h, 2) = (float)clamp((298 * c + 516 * d + 128) >> 8); // blue
            ++w;
            c = y1 - 16;
            t(w, h, 0) = (float)clamp((298 * c + 409 * e + 128) >> 8); // red
            t(w, h, 1) = (float)clamp((298 * c - 100 * d - 208 * e + 128) >> 8); // green
            t(w, h, 2) = (float)clamp((298 * c + 516 * d + 128) >> 8); // blue
            ++w;
        }
    }
}

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-stylefilter", "en-US")

struct style_data
{
    obs_source_t* context;
    bool enabled = false;
    Tensor alpha = Tensor({ 1.f }, Shape(1));
    Tensor style;
};

static const char* style_filter_name(void* unused)
{
    UNUSED_PARAMETER(unused);
    return "Style Filter";
}

static void style_filter_defaults(obs_data_t *settings)
{
    obs_data_set_default_bool(settings, SETTING_ENABLED_NAME, false);
    obs_data_set_default_double(settings, SETTING_ALPHA_NAME, 1.0);
    static string default_style = string(obs_get_module_data_path(obs_current_module())) + "/mosaic.jpg";
    obs_data_set_default_string(settings, SETTING_STYLE_FILE_NAME, default_style.c_str());
}

static void style_filter_update(void* data, obs_data_t* settings)
{
    style_data* filter = (style_data*)data;
    filter->enabled = (bool)obs_data_get_bool(settings, SETTING_ENABLED_NAME);
    filter->alpha(0) = (float)obs_data_get_double(settings, SETTING_ALPHA_NAME);
    string style_image = obs_data_get_string(settings, SETTING_STYLE_FILE_NAME);
    filter->style = LoadImage(style_image, STYLE_IMAGE_WIDTH, STYLE_IMAGE_HEIGHT, STYLE_IMAGE_WIDTH, STYLE_IMAGE_HEIGHT);
}

static void* style_filter_create(obs_data_t* settings, obs_source_t* context)
{
    //style_data* filter = (style_data*)bzalloc(sizeof(style_data));
    style_data* filter = new style_data();

    filter->context = context;
    style_filter_update(filter, settings);

    Tensor::SetForcedOpMode(GPU);

    return filter;
}

static void style_filter_destroy(void* data)
{
    style_data* filter = (style_data*)data;
    delete filter;
    //bfree(data);
}

static obs_properties_t* style_filter_properties(void* data)
{
    obs_properties_t* props = obs_properties_create();
    
    obs_properties_add_bool(props, SETTING_ENABLED_NAME, SETTING_ENABLED_TEXT);
    obs_properties_add_float(props, SETTING_ALPHA_NAME, SETTING_ALPHA_TEXT, 0, 1, 0.01);
    obs_properties_add_path(props, SETTING_STYLE_FILE_NAME, SETTING_STYLE_FILE_TEXT, OBS_PATH_FILE, "*.jpg", nullptr);
    
    return props;
}

static void style_filter_remove(void* data, obs_source_t* parent)
{
    style_data* filter = (style_data*)data;
}

static obs_source_frame* style_filter_video(void* data, obs_source_frame* frame)
{
    style_data* filter = (style_data*)data;
    obs_source_t* parent = obs_filter_get_parent(filter->context);

    if (filter->enabled)
    {
        if (!generator)
        {
            string data_path = string(obs_get_module_data_path(obs_current_module())) + "/";

            input_content = new Placeholder(Shape(frame->width, frame->height, 3), "input_content");
            input_style = new Placeholder(Shape(STYLE_IMAGE_WIDTH, STYLE_IMAGE_HEIGHT, 3), "input_style");
            input_alpha = new Placeholder(filter->alpha, "alpha");

            auto vggModel = VGG19::CreateModel(NCHW, Shape(frame->width, frame->height, 3), false, MaxPool, data_path);
            vggModel->SetTrainable(false);

            vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv1")->Outputs()[0],
                                                 vggModel->Layer("block2_conv1")->Outputs()[0],
                                                 vggModel->Layer("block3_conv1")->Outputs()[0],
                                                 vggModel->Layer("block4_conv1")->Outputs()[0] };

            auto vggEncoder = Flow(vggModel->InputsAt(-1), styleOutputs, "vgg_features");

            auto contentPre = VGG16::Preprocess(input_content, NCHW, false);
            auto stylePre = VGG16::Preprocess(input_style, NCHW, false);

            auto generator = create_generator_model(contentPre, stylePre, input_alpha, vggEncoder);
            
            generator->LoadWeights(data_path + "adaptive_weights.h5", false, true);

            stylized = generator->Outputs()[0];

            /*input = new Placeholder(Shape(frame->width, frame->height, 3), "input");
            generator = create_generator_model(VGG16::Preprocess(input, NCHW));
            stylized = generator->Outputs()[0];

            string data_path = obs_get_module_data_path(obs_current_module());
            generator->LoadWeights(data_path + "/mosaic_weights.h5", false, true);*/
        }

        Tensor frameData(Shape(frame->width, frame->height, 3));
        frameData.OverrideHost();

        if (frame->format == VIDEO_FORMAT_I420)
            I420_2_RGB(frame, frameData);
        else if (frame->format == VIDEO_FORMAT_YUY2)
            YUY2_2_RGB(frame, frameData);

        //frameData.SaveAsImage("e:/_frame.jpg", false);

        //auto results = Session::Default()->Run({ stylized }, { { input_content, &frameData } });
        auto results = Session::Default()->Run({ stylized }, { { input_content, &frameData }, { input_style, &filter->style }, { input_alpha, &filter->alpha } });
        auto frameDataStylized = *results[0];
        //VGG16::DeprocessImage(frameDataStylized, NCHW);

        //frameDataStylized.SaveAsImage("e:/_frame_s.jpg", false);

        if (frame->format == VIDEO_FORMAT_I420)
            RGB_2_I420(frameDataStylized, frame);
        else if (frame->format == VIDEO_FORMAT_YUY2)
            RGB_2_YUY2(frameDataStylized, frame);
    }

    return frame;
}

obs_source_info style_filter = (style_filter = obs_source_info(),
    style_filter.id = "style_filter",
    style_filter.type = OBS_SOURCE_TYPE_FILTER,
    style_filter.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
    style_filter.get_name = style_filter_name,
    style_filter.create = style_filter_create,
    style_filter.destroy = style_filter_destroy,
    style_filter.get_defaults = style_filter_defaults,
    style_filter.update = style_filter_update,
    style_filter.get_properties = style_filter_properties,
    style_filter.filter_video = style_filter_video,
    style_filter.filter_remove = style_filter_remove,
    style_filter
);

bool obs_module_load(void)
{
    obs_register_source(&style_filter);

    return true;
}

void obs_module_unload(void)
{
}
