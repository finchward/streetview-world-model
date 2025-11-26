from dataclasses import dataclass

@dataclass
class Config:
    #Model Loading config
    model_name = "tv6.6"
    load_model = True
    loaded_model = "tv6.6"
    loaded_checkpoint = "main"

    #Sampling config
    is_interacting = True
    interaction_samples = 8
    interactive_model = "tv6.5_b20k"
    interactive_checkpoint = "main.chkp"
    interactive_latent = "idx_35.pt"
    is_latent_init_noise = False
    is_latent_init_zeros = True
    guidance_factor = 1

    #Model architecture config
    layers = 12
    hidden_size = 384
    heads = 6
    patch_size = 4
    attn_dropout = 0.1
    ffd_dropout = 0.1
    mlp_hidden_size_ratio = 4
    memory_layers = 2

    #Training information
    graph_dir = 'graphs'
    sample_every_x_batches = 100 #avoid being divisible by latent_persistence_turns 
    inference_samples = 4
    img_dir = 'images'
    validation_frequency = 200 #sample 1998 and 500 used to be.
    validation_samples = 20
    graph_update_freq = 2
    recent_losses_shown = 50 #used to be 200
    loss_bucket_size = 1 #used to be 100
    save_freq = 250

    #Training
    from_noise = True
    batch_size = 5
    effective_batch_size = 65 #256
    model_resolution = (384, 512)
    latent_persistence_turns = 2
    latent_reset_turns = 4
    shortcut_percent = 0.25
    ema_ratio = 0.975 #0.999x
    max_batches = 500_000
    learning_rate = 1e-4 #1e-4
    weight_decay = 0.1

    #Data generation
    rotation_probability = 0.7
    initial_pages = [
        r"https://www.google.com/maps/@-34.7826845,173.1006281,3a,75y,48.05h,109.63t/data=!3m7!1e1!3m5!1sjWIbsBBMN4YiWwdgLl5Y5A!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-19.63436488427803%26panoid%3DjWIbsBBMN4YiWwdgLl5Y5A%26yaw%3D48.045970941061924!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkzMC4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/@-37.1915928,174.654654,3a,75y,265.22h,85.04t/data=!3m7!1e1!3m5!1sxCmXgULR3Wewk4gGIGPsow!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D4.95869070845373%26panoid%3DxCmXgULR3Wewk4gGIGPsow%26yaw%3D265.2245735720425!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkzMC4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/@-38.2781874,178.2625244,3a,75y,342.95h,88.49t/data=!3m7!1e1!3m5!1sN_l8m4iP1o-8EI-4baOS2g!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D1.5134699273536683%26panoid%3DN_l8m4iP1o-8EI-4baOS2g%26yaw%3D342.9537239759422!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkzMC4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/@-35.0347671,173.9107666,3a,75y,287.18h,96.07t/data=!3m7!1e1!3m5!1s2ZeaVLCvjeBDiQCRYSYttg!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-6.067976971432643%26panoid%3D2ZeaVLCvjeBDiQCRYSYttg%26yaw%3D287.18365123558544!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkzMC4wIKXMDSoASAFQAw%3D%3D",
        r"https://www.google.com/maps/@-34.4401226,172.7131244,3a,75y,247.7h,86.57t/data=!3m7!1e1!3m5!1san-k3n7Q5u_NfQSwzrEpHw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D3.430265383680293%26panoid%3Dan-k3n7Q5u_NfQSwzrEpHw%26yaw%3D247.69731166955927!7i16384!8i8192?entry=ttu&g_ep=EgoyMDI1MDkzMC4wIKXMDSoASAFQAw%3D%3D",
        ] * 3

#     [ 
#         r"https://www.google.com/maps/@-38.5922817,176.8199381,3a,75y,271.3h,104.26t/data=!3m7!1e1!3m5!1s1Pt-bx9x0-vdMyd-R05xZw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-14.260361780245958%26panoid%3D1Pt-bx9x0-vdMyd-R05xZw%26yaw%3D271.30228279891134!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkwNy4wIKXMDSoASAFQAw%3D%3D",
#         r"https://www.google.com/maps/place/Pukaki+Canal+Road,+Canterbury+Region+7999/@-44.3213197,170.0447441,3a,75y,195.92h,96.45t/data=!3m7!1e1!3m5!1sI9B-vhHlUJpLWYoqiLarOw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-6.447370168190574%26panoid%3DI9B-vhHlUJpLWYoqiLarOw%26yaw%3D195.9163922970516!7i16384!8i8192!4m6!3m5!1s0x6d2ae1d22564538b:0xac6cf5a0c84835c2!8m2!3d-44.2098275!4d170.0758422!16s%2Fg%2F11h6yd4mrn?entry=ttu&g_ep=EgoyMDI1MDkwNy4wIKXMDSoASAFQAw%3D%3D",
#         r"https://www.google.com/maps/@-45.1338186,168.7592437,3a,75y,211.88h,90t/data=!3m7!1e1!3m5!1slN5kBn2x9rltM8rmRvc6iw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D0%26panoid%3DlN5kBn2x9rltM8rmRvc6iw%26yaw%3D211.87823!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkwNy4wIKXMDSoASAFQAw%3D%3D",
#         r"https://www.google.com/maps/place/Huia,+Auckland+0604/@-37.0021211,174.5724628,3a,75y,237.14h,110.13t/data=!3m7!1e1!3m5!1sLrys74X-9grhlOM4y6dYqw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-20.131473592702037%26panoid%3DLrys74X-9grhlOM4y6dYqw%26yaw%3D237.13872159890457!7i16384!8i8192!4m6!3m5!1s0x6d0d69b2f8e74d3f:0x500ef6143a2cea0!8m2!3d-37.003466!4d174.5724721!16s%2Fm%2F027k8h0?entry=ttu&g_ep=EgoyMDI1MDkxMC4wIKXMDSoASAFQAw%3D%3D",
#         r"https://www.google.com/maps/@-38.0633252,175.5396599,3a,75y,251.25h,90t/data=!3m7!1e1!3m5!1sDoszpl71ckFRT0ZDix5UHA!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D0%26panoid%3DDoszpl71ckFRT0ZDix5UHA%26yaw%3D251.24602!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkxMC4wIKXMDSoASAFQAw%3D%3D",
#         r"https://www.google.com/maps/@-38.5922817,176.8199381,3a,75y,271.3h,104.26t/data=!3m7!1e1!3m5!1s1Pt-bx9x0-vdMyd-R05xZw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D-14.260361780245958%26panoid%3D1Pt-bx9x0-vdMyd-R05xZw%26yaw%3D271.30228279891134!7i13312!8i6656?entry=ttu&g_ep=EgoyMDI1MDkwNy4wIKXMDSoASAFQAw%3D%3D",
#    ] # make one of these huia

    

    



    