import torch

checkpoint_path = "./checkpoints/stage1/checkpoint_50.pth"
checkpoint = torch.load(checkpoint_path)
patch_embed_layer_state_dict = {}
patch_embed_layer_state_dict['proj.weight'] = checkpoint['model_state_dict']['mp_generator.patch_embed.proj.weight']
patch_embed_layer_state_dict['proj.bias'] = checkpoint['model_state_dict']['mp_generator.patch_embed.proj.bias']
save_path = 'results/model_weights/patch_embed_weights.pth'
torch.save(patch_embed_layer_state_dict, save_path)

