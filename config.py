from dataclasses import dataclass

@dataclass
class Config:
    model_name = "Version_one"
    load_model = False
    loaded_model = "v2"
    loaded_checkpoint = "abc"
    
    is_colab = True
    is_tpu = False
    drive_dir = '/content/drive/ml/streetview'

    sample_every_x_batches = 48 #avoid being divisible by latent_persistence_turns
    inference_samples = 300
    inference_step_size = 1
    img_dir = 'images'
    inference_display_update_freq = 20

    model_resolution = (384, 512)
    features = [64, 128, 256, 512, 1024] 
    #64 -> 128 -> 256 -> 512 -> 1024/16 ->| 2048/16 |  -> 1024/32 append 2048/32 -> 1024/32 -> 512/64 -> 256/128 -> 128/256 -> 64/512
    #256 -> 128 -> 64 -> 32 -> 16 |  192 -> 96 -> 48 -> 24 -> 12
    
    latent_persistence_turns = 8
    predictions_per_image = 1

    group_size = 32
    time_embedding_dim = 512 #Must be even
    movement_embedding_dim = 512
    latent_dimension = 1024
    max_batches = 100_000
    rotation_probability = 0.6
    initial_pages = [ 
        r"https://www.google.com/maps/@-38.5922817,176.8199381,3a,75y,271.3h,104.26t/data=!3m7!1e1!3m5!1s1Pt-bx9x0-vdMyd-R05xZw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-14.260361780245958%26panoid%3D1Pt-bx9x0-vdMyd-R05xZw%26yaw%3D271.30228279891134!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkwNy4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/place/Pukaki+Canal+Road,+Canterbury+Region+7999/@-44.3213197,170.0447441,3a,75y,195.92h,96.45t/data=!3m7!1e1!3m5!1sI9B-vhHlUJpLWYoqiLarOw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-6.447370168190574%26panoid%3DI9B-vhHlUJpLWYoqiLarOw%26yaw%3D195.9163922970516!7i16384!8i8192!4m6!3m5!1s0x6d2ae1d22564538b:0xac6cf5a0c84835c2!8m2!3d-44.2098275!4d170.0758422!16s%2Fg%2F11h6yd4mrn?entry=ttu&g_ep=EgoyMDI1MDkwNy4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/@-45.1338186,168.7592437,3a,75y,211.88h,90t/data=!3m7!1e1!3m5!1slN5kBn2x9rltM8rmRvc6iw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D0%26panoid%3DlN5kBn2x9rltM8rmRvc6iw%26yaw%3D211.87823!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkwNy4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/place/Huia,+Auckland+0604/@-37.0021211,174.5724628,3a,75y,237.14h,110.13t/data=!3m7!1e1!3m5!1sLrys74X-9grhlOM4y6dYqw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-20.131473592702037%26panoid%3DLrys74X-9grhlOM4y6dYqw%26yaw%3D237.13872159890457!7i16384!8i8192!4m6!3m5!1s0x6d0d69b2f8e74d3f:0x500ef6143a2cea0!8m2!3d-37.003466!4d174.5724721!16s%2Fm%2F027k8h0?entry=ttu&g_ep=EgoyMDI1MDkxMC4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/@-38.0633252,175.5396599,3a,75y,251.25h,90t/data=!3m7!1e1!3m5!1sDoszpl71ckFRT0ZDix5UHA!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D0%26panoid%3DDoszpl71ckFRT0ZDix5UHA%26yaw%3D251.24602!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkxMC4wIKXMDSoASAFQAw%3D%3D"
    ] # make one of these huia

    learning_rate = 4e-5
    weight_decay = 0.001

    graph_update_freq = 10
    recent_losses_shown = 1500
    loss_bucket_size = 10
    save_freq = 100



