# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

"""Let's turn off the scientific notation for floating point numbers."""

np.set_printoptions(suppress=True)

"""#Loading and examining the data
We will load our data from a CSV file and put it in a pandas an object of the DataFrame class.
"""

df = pd.read_csv('final_data.csv')

"""Let's take a look at the data:"""

display(df.head(100))

"""We first need to extract our data, from the dataframe, in **NumPy arrays**:

We can use a single **scatter plot** to take a look at our data:
"""

fig = px.scatter(df, x="Temperature", y="Humidity", color="Label")
fig.show()

X = df.drop(['Sensor 1','Sensor 3','Sensor 4','Sensor 7', 'Sensor 9','Sensor 3 Missing','Wind level > 0', 'Wind level > 1', 'Wind level > 2', 'Label','Mode 1', 'Mode 2', 'Mode 3', 'Mode ?'], axis=1).to_numpy()
y_text = df['Label'].to_numpy()
y = LabelEncoder().fit_transform(y_text)

"""As a sanity check, let's check X:"""

X

"""...and the size:"""

X.shape

"""Let's do the same thing for y:"""

y

"""...and for shape of y_text:"""

y.shape

"""#Splitting data
Again, let's split our data into training, validation and test sets. Let's use 60% (90 examples) for training, 20% for validation (30 examples) and the remaining 20% (30 examples) as test data.
"""

(X_train, X_vt, y_train, y_vt) = train_test_split(X, y, test_size=0.4, random_state=0)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_vt, y_vt, test_size=0.5, random_state=0)

"""##Building and visualizing a $k$-NN model"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

"""Next, let's fit our knn to our X_train and y_train. This does nothing but store the training example as $k$-NN is "lazy": It does all calculations as prediction time and by measuring the distance from the operational datapoints provided (whose labels have to be predicted) to each of the training datapoints, finding the closest training datapoints to the operational points, looking at the labels for those closest training datapoints, and finding the majority class among them."""

import pandas as pd

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

knn.fit(X_train, y_train)

"""Now, we can plot our training data in 3D with a **3D scatter plot** (we are going to use surface plots afterwards and the new interface of plotly cannot do surface plots yet, so we are using the older style rather than plotly express):"""

points_colorscale = [
                     [0.0, 'rgb(239, 85, 59)'],
                     [1.0, 'rgb(99, 110, 250)'],
                    ]

layout = go.Layout(scene=dict(
                              xaxis=dict(title='Temperature'),
                              yaxis=dict(title='Humidity'),
                              zaxis=dict(title='Label')
                             ),
                  )

points = go.Scatter3d(x=df['Temperature'], 
                      y=df['Humidity'], 
                      z=y,
                      mode='markers',
                      text=df['Label'],
                      marker=dict(
                                  size=3,
                                  color=y,
                                  colorscale=points_colorscale
                            ),
                     )

fig2 = go.Figure(data=[points], layout=layout)
fig2.show()

"""We have more than 3 features, so let's use a scatter matrix to visualize our data:"""

data_dimensions = df.columns[:5].to_list()

fig = px.scatter_matrix(df, dimensions=data_dimensions, color='Label')
fig.show()

"""###Let's plot the three datasets as well:"""

df_train = pd.DataFrame(np.c_[X_train, y_train], columns=df.columns[:6])
fig2 = px.scatter_matrix(df_train, dimensions=data_dimensions, color='Sensor 3 Missing')
fig2.show()

df_validation = pd.DataFrame(np.c_[X_validation, y_validation], columns=df.columns[:6])
fig3 = px.scatter_matrix(df_validation, dimensions=data_dimensions, color='Sensor 3 Missing')
fig3.show()

df_test = pd.DataFrame(np.c_[X_test, y_test], columns=df.columns[:6])
fig4 = px.scatter_matrix(df_test, dimensions=data_dimensions, color='Sensor 3 Missing')
fig4.show()

"""#Let's try with Decision Tree"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
import pandas as pd
import plotly.express as px
from pydotplus import graph_from_dot_data
from IPython.display import Image

np.set_printoptions(suppress=True)

df = pd.read_csv('final_data.csv')
display(df.head(100))

"""Let's create a **scatter plot** to visualize the data:"""

data_dimensions = df.columns[:-1].to_list()
figure_size = df.shape[1] * 256

fig = px.scatter_matrix(df, dimensions=data_dimensions, color='Label', width=figure_size, height=figure_size)
fig.show()

X = df.drop(['Temperature','Humidity','M1','M2','M3','Sensor 3 Missing','Wind level > 0', 'Wind level > 1', 'Wind level > 2', 'Label','Mode 1', 'Mode 2', 'Mode 3', 'Mode ?'], axis=1).to_numpy()
y = df['Label'].to_numpy()

"""Let's see our data:"""

X

y

"""##Splitting the Data"""

(X_train, X_vt, y_train, y_vt) = train_test_split(X, y, test_size=0.4, random_state=0)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_vt, y_vt, test_size=0.5, random_state=0)

"""##Building and Fitting a Decision Tree"""

from sklearn import tree
dtree = tree.DecisionTreeClassifier()

dtree.fit(X_train, y_train)

"""##Visualizing the decision tree
Now, let's visualize our decision tree. Let's export our model as a special kind of data, create a visual representation form that, generate a graph from that representation and show that graph as an image (it may be a big image, so you may have to scroll to see the whole thing):
"""

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, impurity=False, special_characters=True)
graph = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(), unconfined=True)

"""##Model assessment and selection
So let's evaluate our QuAM. First, let's see what happened on training data.
"""

yhat_train = dtree.predict(X_train)

"""Accuracy Score:"""

accuracy_score(yhat_train, y_train)

"""A perfect 1.0 or 100%! However, that was what was expected given the minimum leaf size was 1, we let the decision tree split leafs with even 2 points in them and we had no tree depth limittation among other things."""

yhat_validation = dtree.predict(X_validation)

accuracy_score(yhat_validation, y_validation)

"""Right. The accuracy on validation data is much lower. Maybe we have overfit to our data. Unrestricted decision trees do that."""

from sklearn import tree
dtree2 = tree.DecisionTreeClassifier(min_samples_leaf = 15)

dtree2.fit(X_train, y_train)

yhat_train2 = dtree2.predict(X_train)

accuracy_score(yhat_train2, y_train)

"""Training accuracy is predictably lower as we restrcited our decision tree, so it does not fit perfectly to training data because of its constraints."""

yhat_validation2 = dtree2.predict(X_validation)

accuracy_score(yhat_validation2, y_validation)

"""Yes! We have a better score on validation points. We may be ovefitting less this time!

##Evaluating the decision tree
Finally, let's use the test data to get a final accuracy performance number for our model. Predict yhat_test2 using dtree2. We can then calculate the accuracy on test data:
"""

yhat_test2 = dtree2.predict(X_test)

accuracy_score(yhat_test2, y_test)

"""The accuracy on test data, on validation data*(0.8164313222079589)* and on training data*(0.8161953727506427)* are close to each other, which is a good sign.

#Conclusion
So, our analysis on the functioning of a specific motor is completed.
We can conclude that the functioning of this motor is devoid of all external factors like **temperature**, **humdity**, **wind level**, and so on. Basically, it depends on the calibration of all the **Sensors** *(1-9 excluding few)* present inside it. We can also see, our model is a success.
"""
