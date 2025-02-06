mod kokoro_core;
mod kokoro_utils;
use kokoro_utils::save_wav;
use kokoro_core::{KokoroInput, KokoroModel, KokoroQuantLevel};

fn main() {
    let voice_name = "af_nova";
    let kokoro_model = KokoroModel::load_model(
        "./models/kokoro-quant-convinteger.onnx",
        KokoroQuantLevel::Fp32,
    );
    let _test = "goodnight everyone, sweet dreams!";
    let _test2 = "fuck all of y'all";
    let input = KokoroInput::new(&_test, voice_name, 1.0);
    let input2 = KokoroInput::new(&_test2, voice_name, 1.0);
    let reshape_result = &kokoro_model.generate(&input).unwrap();
    let reshape_result2 = &kokoro_model.generate(&input2).unwrap();
    save_wav(&reshape_result, 22500, &format!("output{}.wav", 1)).unwrap();
    save_wav(&reshape_result2, 22500, "output2.wav").unwrap();
}
