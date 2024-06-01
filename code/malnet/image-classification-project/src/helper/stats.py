import os

def stats_dataset(directory,file_name = None):
  
    if file_name is not None:
      f = open(file_name, "w")
    else:
      f = None
      
    print("Loading images from", directory,file = f)
    print("Classes found:", os.listdir(directory),file = f)

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)

        # print("Loading images from", class_dir, " "*10, len(os.listdir(class_dir)), "images") 
        # # there must be constant gap between class_dir and number of images


        if os.path.isdir(class_dir):
            count_class = 0
            for family_name in os.listdir(class_dir):
                family_dir = os.path.join(class_dir, family_name)
                print("Images count from", family_dir, " "*(100-len(family_dir)), len(os.listdir(family_dir)), "images")
                count_class += len(os.listdir(family_dir))
            
            print("Total images in",class_dir,":", " "*(100-len(class_dir)),count_class,file = f)
    