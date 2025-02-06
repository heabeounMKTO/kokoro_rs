mod core;
mod kokoro_utils;
use core::{KokoroInput, KokoroModel};
use core::{VOCAB, VOICES};
use kokoro_utils::{kokoro_select_voice, normalize_text, string_to_tokens, KokoroVoice};
use lazy_phonememize::phonememizer::convert_to_phonemes;
use ndarray::{arr2, array, Array, Array2, ArrayD, Dimension};
use ort::inputs;
use ort::value::Tensor;

use hound;
use ndarray::{Array1, ArrayBase, OwnedRepr, Dim};


fn save_wav(data:  &Array1<f32>, sample_rate: u32, filename: &str) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(filename, spec)?;
    
    for &sample in data {
        // Convert float to 16-bit integer
        let scaled = (sample * 32767.0) as i16;
        writer.write_sample(scaled)?;
    }
    
    Ok(())
}

fn main() {
    let voice_name = "af_nova";
    let load = KokoroModel::load_model("./models/kokoro-v0_19.onnx", false);
    println!("load model : {:#?}", load.model.inputs);
    let mut _test = "goodnight everyone, sweet dreams!";
    normalize_text(_test);
    println!("[DEBUG] input TEST {}", _test);
    let input = KokoroInput::new(&_test, voice_name, 1.0);
    println!("onput: {:?}", load.model.outputs);
    let token_tensor =
        ort::value::Tensor::from_array(([1, input.tokens.len()], input.tokens)).unwrap();

    let voice_reshape = input.voice.into_shape_with_order((1, 256)).unwrap();
    let voice_tensor: ort::value::Tensor<f32> =
        ort::value::Tensor::from_array(voice_reshape).unwrap();
    let speed_tensor: ort::value::Tensor<f32> = Tensor::from_array(([1], vec![1.0])).unwrap();

    println!("tken {:?}", token_tensor);
    println!("voice {:?}", voice_tensor);

    let results = load
        .model
        .run(
            inputs!["tokens" => token_tensor, "style" => voice_tensor, "speed" => speed_tensor]
                .unwrap(),
        )
        .unwrap();
    let extract_res = results["audio"]
        .try_extract_tensor::<f32>()
        .unwrap();
    let reshape_result = extract_res.to_owned().into_shape_with_order(extract_res.to_owned().shape()[0]).unwrap(); 
    save_wav(&reshape_result.to_owned(), 22500, "output.wav").unwrap();
    // println!("RES {:?}", extract_res);
}
