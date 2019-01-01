import os

# This python script was used to extract all the nii files from a download from humanconnectome

root_path = "../Aura_Data";
for subdir, dirs, files in os.walk(root_path + "/Healthy"):
    for foldername in dirs:
        if os.path.isdir(root_path + "/Healthy/" + foldername) and foldername[0:6].isdigit():
            # Move preprocessed files
            if "preproc" in foldername:
                if not os.path.isdir(root_path + "/Healthy/NIFTI/" + foldername[0:6]):
                    os.mkdir(root_path + "/Healthy/NIFTI/" + foldername[0:6])
                # Move T1 scan
                try:
                    os.rename(root_path + "/Healthy/" + foldername + "/" + foldername[0:6] + "/MNINonLinear/T1w_restore.1.60.nii.gz" ,
                          root_path + "/Healthy/NIFTI/" + foldername[0:6] + "/T1w_restore.1.60.nii.gz")
                except:
                    print("T1 was already moved.")
                # Move T2 scan
                try:
                    os.rename(root_path + "/Healthy/" + foldername + "/" + foldername[0:6] + "/MNINonLinear/T2w_restore.1.60.nii.gz",
                          root_path + "/Healthy/NIFTI/" + foldername[0:6] + "/T2w_restore.1.60.nii.gz")
                except:
                    print("T2 was already moved.")
            # We assume that unprocessed files have already had a special folder created for them in the
            # NIFTI folder but still check
            else:
                if not os.path.isdir(root_path + "/Healthy/NIFTI/" + foldername[0:6]):
                    os.mkdir(root_path + "/Healthy/NIFTI/" + foldername[0:6])

                for sub, directories, fs in os.walk(root_path + "/Healthy/" + foldername + "/" + foldername[0:6] + "/unprocessed/7T/tfMRI_RETBAR1_AP/"):
                    for f in fs:
                        if "nii" in f:
                            os.rename(root_path + "/Healthy/" + foldername + "/" + foldername[0:6] + "/unprocessed/7T/tfMRI_RETBAR1_AP/" + f,
                                  root_path + "/Healthy/NIFTI/" + foldername[0:6] + "/" + f)

        else:
            print(foldername + " is a file")