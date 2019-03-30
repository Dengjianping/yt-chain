use cmake;

fn main() {
    if cfg!(feature = "cuda") {
        let dst = cmake::Config::new("cuda_mining").build();

        println!("cargo:rustc-link-search=native={}", dst.display());
        println!("cargo:rustc-link-lib=static=cuda_mining");
        
        println!("cargo:rustc-link-lib=dylib=stdc++"); // link to stdc++ lib
        
        let lib_path = env!("LD_LIBRARY_PATH");
        let cuda_lib_path: Vec<_> = lib_path.split(':').into_iter().filter(|path| path.contains("cuda")).collect();
        if cuda_lib_path.is_empty() {
            panic!("Ensure cuda installed on your environment");
        } else {
            println!("cargo:rustc-link-search=native={}", cuda_lib_path[0]);
            println!("cargo:rustc-link-lib=cudart"); // cuda run-time lib
        }
    }
}