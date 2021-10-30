from mmkfeatures.models.image_text_fusion.image_text_feature_wrapper import ImageTextFeaturesWrapper
import torch
if __name__ == "__main__":
    # define a image-text fusion wrapper
    img_txt_feat_wrapper=ImageTextFeaturesWrapper()

    # a json file contains image-text pairs
    json_path="../data/image_text_fusion_data/images_data.json"
    # a folder containing /train and /test image folder
    input_folder="../data/image_text_fusion_data"
    # a folder saving char_embedding and image_conv_models trained in the above process.
    output_folder="../data/image_text_fusion_data/output"

    # parameters settings
    lr=0.001
    epochs=100

    # train a char_embedding model
    img_txt_feat_wrapper.train_char_embedding_model(json_path,output_folder,lr,epochs,"fixed_gru","cvpr","encod_64x64_path")

    # train a image embedding model
    # resetting the torch lib
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    img_txt_feat_wrapper.train_image_conv_model(input_folder,output_folder,lr,epochs)

    # get embedding text for a specific text segment
    embedding_path="../data/image_text_fusion_data/output/char_embedding.pt"
    embedding_text=img_txt_feat_wrapper.get_embedding_text(embedding_path,"girl jumps dutifully. stupid")
    print(embedding_text)

    # get encoder conv given an image
    encoder_path="../data/image_text_fusion_data/output/conv_autoencoder.pt"
    test_image_path="../data/image_text_fusion_data/train/000000.jpeg"
    encode_image=img_txt_feat_wrapper.get_encode_image(encoder_path,test_image_path)
    print(encode_image)




