# VGG-16 layer output channel maps, {layer index : output channels}
vgg_layer_out_c_maps = {
    3 : 64,     # Relu1_2
    8 : 128,    # Relu2_2
    15 : 256,   # Relu3_3
    22 : 512,   # Relu4_3
    29 : 512,    # Relu5_3
    30 : 512   # MaxPool5
}
# VGG-16 layer output size ratio maps, {layer index : output size ratio}
vgg_layer_out_size_ratio_maps = {
    3 : 1.0,      # Relu1_2, (H, W)
    8 : 1/2,      # Relu2_2, (H/2, W/2)
    15 : 1/4,     # Relu3_3, (H/4, W/4)
    22 : 1/8,    # Relu4_3, (H/8, W/8)
    29 : 1/16,     # Relu5_3, (H/16, W/16)
    30 : 1/32    # MaxPool5, (H/32, W/32)
}