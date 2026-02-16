fn main() {
    #[cfg(not(feature = "bindgen"))]
    println!("cargo:rustc-env=WHISPER_DONT_GENERATE_BINDINGS=1");
}
