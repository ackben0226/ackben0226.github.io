---
layout: post
title: Image Search Engine With Deep Learning
image: "/posts/shirts_image.jpeg"
tags: [Python, VGG16, Cosine Similarity, Deep Learning]
---

In this post, we demonstrate an Image Search Engine that uses Convolutional Neural Networks (CNNs) and deep feature extraction to find visually similar images. The application is useful for domains such as fashion, e-commerce, or photography, where visual similarity plays a crucial role.
<br/> We leverage VGG16, a pre-trained deep learning models, which extracts rich image embeddings and compares them using similarity metrics. This enables users to upload a query image and instantly receive the top visually similar images from a reference database.

## Project Overview
Our client had been analysing their customer feedback, and one thing in particular came up a number of times.
<br/> Their customers are aware that they have a great range of competitively priced products in the clothing section - but have said they are struggling to find the products they are looking for on the website.
<br/> They are often buying much more expensive products, and then later finding out that we actually stocked a very similar, but lower-priced alternative.

Based upon our work for them using a Convolutional Neural Network, they want to know if we can build out something that could be applied here.

## Key Features
- _Deep feature extraction using VGG16_
- _Search engine returns top-k similar images_
- _Works with any custom image dataset (e.g., footwear)_
- _Cosine distance metric for similarity computation_
- Uses pre-trained weights (transfer learning)
- _Fast approximate search with efficient vector operations_
- _Built with TensorFlow, Keras, scikit-learn, and NumPy_

## Project Structure
```text
Image-Search-Engine/
│
├── model/                      # Trained model artifacts
│   └── feature.pkl             # Precomputed CNN feature vectors (serialized)
│
├── images/                     # Image dataset directory
│   └── footwear_dataset/       
│       ├── heels/
│       ├── sneakers/
│       └── boots/
│
├── Image Search Engine-Deep Learning.ipynb  # End-to-end pipeline (EDA, modeling, evaluation)
├── app.py                      # Web application interface (Flask/Streamlit)
└── README.md                   # Project documentation (setup, usage, technical details)
```
## Setting Up VGG16
Keras makes the use of VGG16 very easy. We download the bottom of the VGG16 network (everything up to the Dense Layers) and then add a parameter to ensure that the final layer is not a Max Pooling Layer but instead a Global Max Pooling Layer

In the code below, we:

Import the required packaages
Set up the image parameters required for VGG16
Load in VGG16 with Global Average Pooling
Save the network architecture & weights for use in search engine

```ruby
# Import packages
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle

# Image target
img_width = 224
img_height = 224
num_channels = 3

# Image architecture
# Load VGG16 without the top classification layer
vgg = VGG16(input_shape = (img_width, img_height, num_channels), include_top=False, pooling = 'avg')
# vgg = VGG16(input_shape = (224, 224, 3), include_top=False, pooling = 'avg')
```

## Image Preprocessing
After importing the libraries, all images are preprocessed to meet the input requirements of the CNN model (224x224 resolution), including _resizing to 224×224 pixels_, _normalization using model-specific preprocessing_ and _batch dimension expansion_.

The preprocessing pipeline is implemented as follows:

```python
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))      # Load and resize image
    image = img_to_array(image)                               # Convert to numpy array
    image = np.expand_dims(image, axis=0)                     # Add batch dimension
    image = preprocess_input(image)                           # Apply model-specific preprocessing
    
    return image
```

## Feature Extraction
In image search engine, simple pixel-level does not capture a high-level visual features and semantic meaning unlike the feature vectors. In this case, we use a pre-trained VGG16 model (excluding the top classification layers) to extract 4096-dimensional feature vectors from each image. These feature vectors represent high-level visual characteristics captured by the model and enables semantic similarity comparisons than basic pixel-level matching. This approach is crucial for accurately identifying and retrieving similar images.
 
```ruby
# featurise image function/feature extraction
def featurise_image(image):
    featurise_vector =  model.predict(image)
    return featurise_vector
```

## Feature Storage
To enable fast and efficient image similarity search, we serialize and store the extracted feature vectors in Pickle format (.pkl). This approach offers significant performance benefits by eliminating the need to recompute embeddings for every query. Pickle provides a compact binary format that allows for rapid deserialization of large volumes of feature data into memory during inference. By storing precomputed vectors, we perform the expensive embedding computation only once. As the image dataset scales, this method ensures efficient and scalable querying without the overhead of reprocessing images repeatedly.

```ruby
# Save key objects for future use
with open('model/file_name.pkl', 'wb') as file:
    pickle.dump(file_name, file)

# Save of features
with open('model/feature_list.pkl', 'wb') as file:
    pickle.dump(feature_list, file)
```

## Image Search
Once the feature vector pickle files are loaded using (__feature_list = pickle.load(file)__), we submit a query image. This image is then transformed into a feature vector and compared against stored vectors (embeddings) using cosine similarity. 

### Setup
In the code below, we:
- Load in our VGG16 model
- Load in our filename store & feature vector store
- Specify the search image file
- Specify the number of search results we want

```python
# Load in the files (required object)
model = load_model('model/vgg16_model.keras', compile = False)

# Read file_name saved
# file_name = pickle.load(open('model/file_name.pkl', 'rb')

with open('model/file_name.pkl', 'rb') as file:
    file_name = pickle.load(file)

# Read feature_list saved
# feature_list = pickle.load(open('model/feature_list.pkl', 'rb')

with open('model/feature_list.pkl', 'rb') as file:
    feature_list = pickle.load(file)
```
The system then retrieves and displays the top 5 to 9 most similar images based on semantic content, enabling real-time, high-accuracy image search with minimal latency—essential for a responsive user experience at

```ruby
# Search parameters
search_results_n = 10  # Show to 10 closest matches
search_image = '000001.jpg'
all_image = listdir(image_path)
search_image = os.path.join(image_path, all_image[0])
print("First image path:", search_image)
```

<br/><img src="https://github.com/user-attachments/assets/14fa3834-48b8-41ef-981c-45a5724c2b66" alt="000001" width="200"/>


## Using Cosine Similarity to Locate Similar Images
To locate images from our dataset of 112 images (stored in _image_path_) that are similar to our given __search image__ above, we need to compare the feature vector of the given image to the feature vectors of all our base images. We then use the _**NearestNeighbors**_ class from _**scikit-learn**_ and then apply the _**Cosine Distance metric**_ to calculate the angle of difference between the feature vectors.
<br/> Cosine Distance essentially measures the angle between any two vectors, and it looks to see whether the two vectors are pointing in a similar direction or not. The more similar the direction the vectors are pointing, the smaller the angle between them in space and the more different the direction the LARGER the angle between them in space. This angle gives us our cosine distance score.

By calculating this score between our search image vector and each of our base image vectors, we can be returned the images with the eight lowest cosine scores - and these will be our eight most similar images, at least in terms of the feature vector representation that comes from our VGG16 network!

We use the code below to instantiate the Nearest Neighbours logic and specify our metric as Cosine Similarity after preprocessing & featurising logic to the search image.

```ruby
# Preprocess and feature search image
preprocessed_image = preprocess_image(search_image)
      
search_feature_vector = featurise_image(preprocessed_image)

# Instantiate nearest neighbors
image_neighbors = NearestNeighbors(n_neighbors = search_results_n, metric = 'cosine')
image_neighbors.fit(feature_list)

# For search image
image_distances, image_indices = image_neighbors.kneighbors(search_feature_vector)

# Convert closest image indices & distances to lists
image_indices = image_indices[0].tolist()
image_distances = image_distances[0].tolist()

# Get list of filenames from search result
search_result_files = [file_name[i] for i in image_indices]
```
## Cosine Similarity Plot
We use the following syntax to visualize the distribution of cosine distances between the search image and its top image matches in the image similarity search.
This histogram helps interpret how closely related the top retrieved images are to the query image:
- A tight cluster of low cosine distances indicates strong similarity across the top matches.
- A wider spread suggests more variation in similarity, which could reflect less consistency in the retrieved images.A tight cluster of low distances indicates high similarity, while more spread-out bars suggest variability.

```python
def plot_cosine_distance_distribution(image_distances):
    plt.hist(image_distances[:9], bins=10, color='salmon', edgecolor='black')
    plt.title("Cosine Distance Distribution of Top Matches")
    plt.xlabel("Cosine Distance")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
```
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/995e0dc3-ee13-441e-b091-9153c8a3a273"/>


## Plot Search Results
We now have all of the information about the 8 most similar images to our search image - let’s see how well it worked by plotting those images!

We plot them in order from most similar to least similar, and include the cosine distance score for reference (smaller is closer, or more similar)

```ruby
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

for counter, result_file in enumerate(search_result_files[:9]):
    # Open with PIL first to resize
    img = Image.open(result_file)
    img.thumbnail((1024, 1024))  # Resize to max 1024x1024 while keeping aspect ratio
    
    # Then convert to array
    plt.figure(figsize=(18, 8))
    img = img_to_array(img).astype('uint8') / 255.0
    ax = plt.subplot(3, 3, counter + 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(img)
    
    # Add similarity score (e.g., cosine distance)
    score = image_distances[counter] 
    plt.text(
        0.5, -0.1,
        f"Score: {round(score, 3)}",
        fontsize=9,
        ha='center',
        transform=ax.transAxes
    )
    
plt.tight_layout()
plt.show()
```
<img width="159" height="222" alt="image" src="https://github.com/user-attachments/assets/bf9a113a-8c37-4050-892d-afae2b691581"/><img width="201" height="222" alt="image" src="https://github.com/user-attachments/assets/d0a40974-cc74-4a83-b67c-464c1d82bb9f"/><img width="154" height="222" alt="image" src="https://github.com/user-attachments/assets/7ba98de6-7be1-4a2f-a411-7ad1cdec7882" /><img width="140" height="222" alt="image" src="https://github.com/user-attachments/assets/a87c2b4d-7259-4b1a-8d0a-7191b93c210c" /><img width="201" height="222" alt="image" src="https://github.com/user-attachments/assets/81b7f1cf-8366-4ef2-90cf-f2ffa45967b1" /><img width="201" height="222" alt="image" src="https://github.com/user-attachments/assets/2191b6fe-0552-43a7-b90d-d68e1123c465" /><img width="201" height="222" alt="image" src="https://github.com/user-attachments/assets/dbb5cd9d-e514-40d3-9a95-2bdf0013889e" /><img width="154" height="222" alt="image" src="https://github.com/user-attachments/assets/b32fa92b-dc67-467d-9139-c54d837beef4" /><img width="231" height="222" alt="image" src="https://github.com/user-attachments/assets/36ccb83d-ac5c-4fc9-b0c3-a6ca32958654" />








