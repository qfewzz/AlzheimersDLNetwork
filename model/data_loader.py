import os
import sys
from typing import TYPE_CHECKING
from torch.utils.data import Dataset

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, parent_dir)
# import const

if not TYPE_CHECKING:
    const = sys.modules['AlzheimersDLNetwork.const']

if TYPE_CHECKING:
    import const
    

from . import data_loader_utils

class MRIData(Dataset):
    """
    MRI data
    The dictionaries AD_Img_Dict.pkl and MCI_Img_Dict.pkl contain key-value
      pairs of the following:
      key:      subject ID
      value:    paths to relevant images (i.e. multiple images per subject)
    These dictionaries are converted to arrays and passed into this dataset,
    where the paths will be accessed and their neuroimages processed into tensors.
    """

    def __init__(self, root_dir, data_array):
        """
        Args:
            root_dir (string): directory of all the images
            data_array (list): array that contains one [key, [value]*] entry for each patient,
                               where key:       subject ID
                                     value:     paths to patient's MRI .nii neuroimages
        """
        self.root_dir = root_dir
        self.data_array = data_array
        self.print_on = True

    def __len__(self):
        """
        Returns length of dataset       (required by DataLoader)
        """
        return len(self.data_array)  # the number of patients in the dataset
    

    def __getitem__(self, index):
        """
        Allows indexing of dataset      (required by DataLoader)
        Returns a tensor that contains the patient's MRI neuroimages and their diagnoses (AD or MCI)
        """
        current_patient_images_label = self.data_array[index]

        return data_loader_utils.get(
            self.root_dir,
            current_patient_images_label,
            index,
            len(self.data_array),
        )

    
