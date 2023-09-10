
# cm = torch.zeros(num_classes,num_classes) 

#         # Confusion matrix
#         # y_true = torch.squeeze(val_labels[0]).detach()
#         # y_true = y_true.reshape(num_classes, -1)
#         # y_true = np.argmax(y_true, axis = 0)

#         # y_pred = torch.squeeze(val_outputs[0]).detach()
#         # y_pred = y_pred.reshape(num_classes, -1)
#         # y_pred = np.argmax(y_pred, axis = 0)

#         # cm += confusion_matrix(y_true, y_pred, labels=range(num_classes))

#                             # Hausdorff distance
#         # hd += compute_hausdorff_distance(y_pred = val_outputs[0], y = val_labels[0], include_background=False, distance_metric='euclidean', spacing=[0.40519333, 0.40519333, 0.625])


# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
# for i, val_data in enumerate(val_loader):
# roi_size = (96, 96, 96)
# sw_batch_size = 4
# val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
# # plot the slice [:, :, 80]
# plt.figure("check", (18, 6))
# plt.subplot(1, 3, 1)
# plt.title(f"image {i}")
# plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
# plt.subplot(1, 3, 2)
# plt.title(f"label {i}")
# plt.imshow(val_data["label"][0, 0, :, :, 80])
# plt.subplot(1, 3, 3)
# plt.title(f"output {i}")
# plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])

# #plt.show()
# plt.savefig(root_dir + f'Example predictions {i}.png')

# # Test images
# test_images = sorted(glob.glob(os.path.join(data_dir, "test/imagesTs", "*.nii.gz")))
# test_data = [{"image": image} for image in test_images]

# test_transforms = Compose(
# [
# LoadImaged(keys=["image"]),
# EnsureChannelFirstd(keys=["image"]),
# ScaleIntensityRanged(
# keys=["image"],
# a_min=-923,
# a_max=870,
# b_min=-923,
# b_max=870,
# clip=True,
# ),
# NormalizeIntensityd(
# keys=["image"],
# subtrahend = 178.34890741220008,
# divisor = 27.630870194013152,
# ),
# Orientationd(keys=["image"], axcodes="RAS"),
# Spacingd(keys=["image"], pixdim=(0.36816406, 0.36816406, 0.40000001), mode=("bilinear")),
# ]
# )

# test_ds = Dataset(data=test_data, transform=test_transforms)
# test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

# post_transforms = Compose(
# [
# Invertd(
# keys="pred",
# transform=test_transforms,
# orig_keys="image",
# meta_keys="pred_meta_dict",
# orig_meta_keys="image_meta_dict",
# meta_key_postfix="meta_dict",
# nearest_interp=True,
# to_tensor=True,
# ),
# # AsDiscreted(keys="pred", argmax=True, to_onehot=3),
# ]
# )

# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()

# with torch.no_grad():
# for test_data in test_loader:
# test_inputs = test_data["image"].to(device)
# roi_size = (96, 96, 96)
# sw_batch_size = 4
# test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

# test_data = [post_transforms(i) for i in decollate_batch(test_data)]

# volume = torch.argmax(test_data[0]["pred"], dim=0).detach().cpu()

# image = nib.load(test_inputs.meta["filename_or_obj"][0])

# nifti_volume = nib.Nifti1Image(volume, image.affine, image.header)
# filename = test_inputs.meta["filename_or_obj"][0].split('/')[-1]

# pred_filename = filename.split('.')[0] + '_seg_out.nii.gz'
# nib.save(nifti_volume, pred_dir + pred_filename)

# # TODO: Save individual class dice metrics on held-out test, specific to each sample

# # Uncomment the following lines to visualize the predicted results
# # test_output = from_engine(["pred"])(test_data)
# # loader = LoadImage()

# # original_image = loader(test_output[0].meta["filename_or_obj"])

# # plt.figure("check", (18, 6))
# # plt.subplot(1, 2, 1)
# # plt.imshow(original_image[:, :, 20], cmap="gray")
# # plt.subplot(1, 2, 2)
# # plt.imshow(test_output[0].detach().cpu()[1, :, :, 20])
# # plt.show()
