import nibabel as nib
import matplotlib.pyplot as plt

epi_img = nib.load('../Aura_Data/Healthy/NIfTI/100610/T1w_restore.1.60.nii.gz')
epi_img_data = epi_img.get_fdata()
print(epi_img_data.shape)
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[112, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
print("Showing slices...")
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()