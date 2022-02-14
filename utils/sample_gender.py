from shutil import copyfile
from random import seed,choice
import random
import argparse
import os
from sklearn.utils import resample

def main(base_dir, sample_type, out_dir, model_var):
    """
    Sample a dataset and save the samples in a new folder.
    Args:
        base_dir(str) - Directory containing data to be sampled.
        sample_type(train,test,val) - Type of data being sampled.
        out_dir(str) - Directory to save sampled data.
        model_var(single,GEI,keyframe) - Model variation being trained. 
    """

    #Set up male and female input paths
    m_in_dir = os.path.join(base_dir,"male")
    f_in_dir = os.path.join(base_dir, "female")

    print (m_in_dir)
    print (f_in_dir)

    #Set up male and female output paths
    m_out_dir = os.path.join(out_dir, sample_type, "male")
    f_out_dir = os.path.join(out_dir, sample_type, "female")

    #get data
    male_data = get_dataset(m_in_dir,model_var)
    female_data = get_dataset(f_in_dir,model_var)

    #Find min size of dataset and set as no_samples, effectively downsampling larger of the two.
    no_samples = min(len(male_data),len(female_data))
    print("Number of samples: ", no_samples)

    #Sample for male and female
    sample(male_data,no_samples,m_out_dir)
    sample(female_data,no_samples,f_out_dir)

def get_dataset(in_dir,model_var):
    """
    Get all datapoints at the lowest point of the input directory tree.
    Args:
        in_dir(str)- Input directory tree.
        model_var(str) - Model variation: single,GEI or keyframe.
    Return:
        sils/GEIs/keyframes(list) - returns a list of the datapoints from the selected variation.
    """
    lowest_dirs = []

    variations = ["nm-01","nm-02","nm-03","nm-04","nm-05","nm-06"]

    #Single Frame
    if model_var == "single":
        sils = []
        #get all files in dataset to be sampled 
        for root,dirs,files in os.walk(in_dir):
            if files and not dirs:
                if any(var in root for var in variations):
                    for f in files:
                        sils.append(root + "/"+ f)
        
        return sils
        
    #GEI
    elif model_var == "GEI":
        GEIs = []
        #get all files in dataset to be sampled 
        for root,dirs,files in os.walk(in_dir):
            if files and not dirs:
                if any(var in root for var in variations):
                    for f in files:
                        GEIs.append(root + "/"+ f)

        return GEIs

    #Keyframes
    elif model_var == "keyframe":
        keyframes = []
        #get all files in dataset to be sampled 
        for root,dirs,files in os.walk(in_dir):
            #print(root)
            if files and not dirs:
                if any(var in root for var in variations):
                    for f in files:
                        #print(f)
                        if f == "stack_mean_keyframe.png":
                            keyframes.append(root + "/"+ f)

        return keyframes


def sample(data, no_samples, out_dir):
    """
    Randomly select a sample from the input data.
    Args:
        in_dir(str) - Directory containing data to be sampled.
        no_samples(int) - Number of samples to save.
        out_dir(str) - Directory to save samples.
    """

    #Set seed to sample size to ensure data is consistent when
    # same size dataset is sampled on a set of data.
    random.seed(no_samples)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    #get samples
    samples = random.sample(data, no_samples)

    #Copy samples to out directory
    for i,s in enumerate(samples):
        #print (i, s)
        outname = os.path.join(out_dir,str("%03d.png" % (i + 1)))
        copyfile(s, outname)


if __name__=='__main__':
 
    parser = argparse.ArgumentParser(description='Sample data for gender classification.')
     
    parser.add_argument(
        'base_dir',
        help="Directory that contains the data to be sampled.")   
        
    parser.add_argument(
        'sample_type',
        help="Sample type: train, val or test.")   
    
    parser.add_argument(
        'out_dir',
        help="Directory to save samples.")

    parser.add_argument(
            'model_var',
            choices=("single","GEI","keyframe"),
            default="single",
            help="Model variation being trained. single, GEI, or keyframe.")

    args = parser.parse_args()

    main(args.base_dir, args.sample_type, args.out_dir, args.model_var) 
