# kNN_From_Scratch

I implemented the k nearest neighbors (kNN) classification algorithm on python. This algorithm is used to predict the classes of new data points using the known data points. We get the coordinates of the new data points and calculate the distances between the new data points and all of the known ones. As the next step, we sort the distances and get ‘k’ number of the closest points. If we were using uniform weights, we would simply count the number of points for each class and the class with the highest number would be our prediction for the new point. However, in this project I used inverse-distance weights, instead of simply counting the points we give weights for each point. These weights are inversely correlated with the distances, so a closer point gets assigned a higher weight.

# Training Data

<img width="433" alt="plot" src="https://user-images.githubusercontent.com/54302889/145944753-31647782-fd6b-48c2-984c-87df95ea2318.png">

# Infinite Weights Problem

This algorithm created a problem when the test data and the training data had a point with the same coordinates, because the distance between them would be zero. This would give the point an infinite weight and the other points wouldn’t matter at all. To solve this, I increased the distance to 0.01 whenever the actual distance was zero or lower than 0.01. I decided to use this number because of the dataset that was used. If I have chosen a larger number, there would be lots of points with the same edited distance. If I have chosen a smaller number, points with the same coordinates would get way too heavy weights making it almost the same as the infinite weights problem and this could cause overfitting since an out-of-place point in the training data would influence a significant area in the decision boundary.

# Classification Results

<img width="498" alt="image" src="https://user-images.githubusercontent.com/54302889/145945561-51c5252e-b910-4159-8f0c-0abb66c4265a.png">

# Decision Boundaries

<img width="903" alt="image" src="https://user-images.githubusercontent.com/54302889/145945717-f9b10cfa-cc14-4fd9-81f6-b9805fd37fc5.png">

<br>

<img width="875" alt="image" src="https://user-images.githubusercontent.com/54302889/145945758-75ccd3f7-95e7-45c2-8a4e-dd1c51cad4aa.png">
