from flask import Flask, request, jsonify, render_template, redirect, flash, send_file, Response
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.metrics import accuracy_score,mean_squared_error,recall_score,confusion_matrix,f1_score,roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import top_k_accuracy_score,r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
import warnings
warnings.filterwarnings("ignore") 

mpl.rcParams["figure.figsize"] = [7, 7]
mpl.rcParams["figure.autolayout"] = True

app = Flask(__name__)

path = "supply_chain_data.csv"
MPSCData=pd.read_csv(path)

def transform_data():
      MPSCData.columns = [col.lower().replace(' ', '_') for col in MPSCData.columns]
      MPSCData.rename(columns=lambda x: x.replace("(", "").replace(")", ""), inplace=True)
      MP_SC = MPSCData.copy()
      le = preprocessing.LabelEncoder()# create the Labelencoder object
      MP_SC['product_type']= le.fit_transform(MP_SC['product_type'])#convert the categorical columns into numeric
      MP_SC['customer_demographics']= le.fit_transform(MP_SC['customer_demographics'])
      MP_SC['shipping_carriers']= le.fit_transform(MP_SC['shipping_carriers'])
      MP_SC['location']= le.fit_transform(MP_SC['location'])
      MP_SC['sku']= le.fit_transform(MP_SC['sku'])
      MP_SC['inspection_results']= le.fit_transform(MP_SC['inspection_results'])
      MP_SC['transportation_modes']= le.fit_transform(MP_SC['transportation_modes'])
      MP_SC['routes']= le.fit_transform(MP_SC['routes'])
      MP_SC['supplier_name']= le.fit_transform(MP_SC['supplier_name'])
      return MP_SC


def makeplot():
      i=0
      MP_SC = transform_data()
      SC_features=MP_SC[['product_type','price',
       'number_of_products_sold', 'revenue_generated', 'customer_demographics', 'lead_times', 'order_quantities', 'shipping_times',
        'shipping_costs', 'location', 'stock_levels',
       'lead_time', 'production_volumes', 'manufacturing_lead_time',
       'manufacturing_costs', 'inspection_results', 'defect_rates',
       'transportation_modes', 'routes', 'costs']]
      fig = plt.figure(figsize=(12,7.5))
      sns.heatmap(SC_features.corr(), annot = True, fmt = '.2f', cmap = "RdYlGn")
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      MP=MP_SC[['product_type','price',
       'number_of_products_sold', 'revenue_generated', 'customer_demographics', 'lead_times', 'order_quantities', 'shipping_times',
        'shipping_costs', 'location', 'stock_levels',
       'lead_time', 'production_volumes', 'manufacturing_lead_time',
       'manufacturing_costs', 'inspection_results', 'defect_rates',
       'transportation_modes', 'routes', 'costs']]
      sns.set_style('whitegrid')
      sns.pairplot(MP,height = 1.25)
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      STOCK=MP_SC[['price',
       'number_of_products_sold', 'revenue_generated', 'order_quantities',
        'shipping_costs', 'stock_levels',
       'lead_time', 'production_volumes', 'manufacturing_lead_time',
       'manufacturing_costs', 'defect_rates']]
      sns.pairplot(STOCK,height = 1.25)
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      COSTS=MP_SC[['price',
       'number_of_products_sold', 'revenue_generated', 'lead_times', 'order_quantities',
        'shipping_costs', 'stock_levels',
       'lead_time', 'production_volumes', 'manufacturing_lead_time',
       'manufacturing_costs', 'defect_rates', 'costs']]
      sns.pairplot(COSTS,height = 1.25)
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      Product = MPSCData.groupby('product_type') 
      Route = MPSCData.groupby('routes')
      Customer=MPSCData.groupby('customer_demographics')
      Shipping=MPSCData.groupby('shipping_carriers')
      Location=MPSCData.groupby('location')
      Transportation=MPSCData.groupby('transportation_modes')

      dvg1 = plt.figure(figsize=(2,2))
      plt.subplot(3, 2, 1)
      Product['stock_levels'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Total Stock")

      plt.subplot(3, 2, 2)
      Product['order_quantities'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Total Order")

      plt.subplot(3, 2, 3)
      Product['manufacturing_costs'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Manufacturing Costs")

      plt.subplot(3, 2, 4)
      Product['revenue_generated'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Revenue")

      plt.subplot(3,2,5)
      data_Products=MPSCData.groupby(['product_type'])['sku'].count().reset_index(name='number_of_products_sold').sort_values(by= 'number_of_products_sold', ascending= False)
      plt.pie(data_Products['number_of_products_sold'],labels = data_Products['product_type'])
      plt.tight_layout()
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      dvg2 = plt.figure(figsize=(6,7))
      data_Customers=MPSCData.groupby(['customer_demographics'])['sku'].count().reset_index(name='number_of_products_sold').sort_values(by= 'number_of_products_sold', ascending= False)
      plt.pie(data_Customers['number_of_products_sold'],labels = data_Customers['customer_demographics'])
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      Customer_Segment_by_Products= MPSCData.groupby(["customer_demographics","product_type"])["sku"].count().reset_index()
      dvg3 = plt.figure(figsize=(6,7))
      sns.barplot(Customer_Segment_by_Products, x = 'customer_demographics',y = 'sku',hue = 'product_type')
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      dvg4 = plt.figure(figsize=(8,8))
      plt.subplot(3, 2, 1)
      Route['revenue_generated'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Revenue")

      plt.subplot(3, 2, 2)
      Route['order_quantities'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Total Order")

      plt.subplot(3, 2, 3)
      Route['costs'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Tranportation Costs")

      plt.subplot(3, 2, 4)

      Route['shipping_times'].mean().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Avg.Shipping Time")

      plt.subplot(3,2,5)
      data_Routes=MPSCData.groupby(['routes'])['sku'].count().reset_index(name='number_of_products_sold').sort_values(by= 'number_of_products_sold', ascending= False)
      plt.pie(data_Routes['number_of_products_sold'],labels = data_Routes['routes'])
      plt.tight_layout()
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      dvg5 = plt.figure(figsize=(8,8))
      plt.subplot(3, 2, 1)
      Transportation['revenue_generated'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Revenue")

      plt.subplot(3, 2, 2)
      Transportation['order_quantities'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Total Order")

      plt.subplot(3, 2, 3)
      Transportation['costs'].sum().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Tranportation Costs")

      plt.subplot(3, 2, 4)

      Transportation['shipping_times'].mean().sort_values(ascending=False).plot.bar(figsize=(12,14), title="Avg. Shipping Time")

      plt.subplot(3,2,5)
      data_Transportation=MPSCData.groupby(['transportation_modes'])['sku'].count().reset_index(name='number_of_products_sold').sort_values(by= 'number_of_products_sold', ascending= False)
      plt.pie(data_Transportation['number_of_products_sold'],labels = data_Transportation['transportation_modes'])
      plt.tight_layout()
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      Routes_by_Transportation= MPSCData.groupby(["routes","transportation_modes"])["sku"].count().reset_index()
      dvg6 = plt.figure(figsize=(6,7))
      sns.barplot(x = Routes_by_Transportation['routes'],y = Routes_by_Transportation['sku'],hue = Routes_by_Transportation['transportation_modes'])
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      train_MPSC = MP_SC.copy()
      train_MPSC['lead_times']=train_MPSC['lead_times'].astype(int)
      train_MPSC['shipping_times']= train_MPSC['shipping_times'].astype(int)
      train_MPSC['lead_time']= train_MPSC['lead_time'].astype(int)
      train_MPSC['manufacturing_lead_time']= train_MPSC['manufacturing_lead_time'].astype(int)
      X=train_MPSC[[ 'sku', 'price',
        'revenue_generated',
        'lead_times', 'shipping_times', 'shipping_costs',
       'lead_time', 'production_volumes', 'manufacturing_lead_time',
       'manufacturing_costs', 'inspection_results', 'defect_rates', 'routes', 'costs']]
      Y=train_MPSC[['stock_levels']]
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
      model1 = LinearRegression()
      model1.fit(X_train, Y_train)
      Y_pred = model1.predict(X_test)
      lm1g = plt.figure(figsize=(6,6))
      plt.scatter(Y_test, Y_pred)
      sns.regplot(x=Y_test, y=Y_pred)
      plt.xlabel('Actual Stocks Levels')
      plt.ylabel('Predicted Stocks Levels')
      plt.title('Predicted vs Actual Stocks Levels')
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      train_MPSC['Intercept'] = 1
      X=train_MPSC[['Intercept', 'sku', 'lead_times', 'order_quantities', 'location']]
      Y=train_MPSC[['costs']]
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
      model2 = LinearRegression()
      model2.fit(X_train, Y_train)
    
      Y_pred = model2.predict(X_test)
      lm2g = plt.figure(figsize=(6,6))
      plt.scatter(Y_test, Y_pred)
      sns.regplot(x=Y_test, y=Y_pred)
      plt.xlabel('Actual Transportation Costs')
      plt.ylabel('Predicted Transportation Costs')
      plt.title('Predicted vs Actual Transportation Costs')
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      train_MP=MP_SC.copy()
      # Create feature and target arrays
      X = train_MP[['product_type', 'sku', 'price', 'availability', 'number_of_products_sold', 'revenue_generated', 'customer_demographics', 'stock_levels', 'lead_times', 'order_quantities', 'shipping_times', 'shipping_carriers', 'shipping_costs', 'supplier_name', 'location', 'lead_time', 'production_volumes', 'manufacturing_lead_time', 'manufacturing_costs', 'inspection_results', 'defect_rates', 'transportation_modes', 'costs']]
      y = train_MP[['routes']]
 
    
      # Split into training and test set
      X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)


      neighbors = np.arange(1, 31)
      train_accuracy = np.empty(len(neighbors))
      test_accuracy = np.empty(len(neighbors))
  
      # Loop over K values
      for j, k in enumerate(neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train.values.ravel())
      
            # Compute training and test data accuracy
            train_accuracy[j] = knn.score(X_train, y_train.values.ravel())
            test_accuracy[j] = knn.score(X_test, y_test.values.ravel())

      dvg10 = plt.figure(figsize=(6,7))
      plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy', color="purple")
      plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy', color="green")
  
      plt.legend()
      plt.xlabel('n_neighbors')
      plt.ylabel('Accuracy')
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      knn = KNeighborsClassifier(n_neighbors=30)
 
      #Fit the model
      knn.fit(X_train,y_train.values.ravel()) 
      #Get accuracy. 
      #Note: In case of classification algorithms score method 
      #represents accuracy. 
      knn.score(X_test,y_test.values.ravel())

      X = train_MP[['product_type', 'sku', 'price', 'availability', 'number_of_products_sold', 'revenue_generated', 'customer_demographics', 'stock_levels', 'lead_times', 'order_quantities', 'shipping_times', 'shipping_carriers', 'shipping_costs', 'supplier_name', 'location', 'lead_time', 'production_volumes', 'manufacturing_lead_time', 'manufacturing_costs', 'inspection_results', 'defect_rates', 'transportation_modes', 'costs']]
      y = train_MP[['routes']]

  
      # Split into training and test set
      X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)


      knn = KNeighborsClassifier(n_neighbors=10)
      knn.fit(X_train, y_train.values.ravel())
  
      # Calculate the accuracy of the model
      y_pred = knn.predict(X_test)
      #print(confusion_matrix(y_test.values.ravel(), y_pred))

      k_values = [j for j in range (1,31)]
      scores = []

      scaler = StandardScaler()
      X = scaler.fit_transform(X)

      for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, X, y.values.ravel(), cv=5)
            scores.append(np.mean(score))

      dvg11 = plt.figure(figsize=(6,7))
      sns.lineplot(x = k_values, y = scores, marker = 'o')
      plt.xlabel("K Values")
      plt.ylabel("Accuracy Score")
      i=i+1
      plt.savefig(f'static/images/try{i}.png')


@app.route('/')

@app.route('/index')
def index():
      return render_template('index.html')

@app.route('/contact')
def contact():
      return render_template('contact.html')

@app.route('/about')
def about():
      return render_template('about.html')

@app.route('/prediction')
def prediction():
      return render_template('prediction.html')

@app.route('/blog')
def blog():
      return render_template('blog.html')

@app.route('/login')
def login():
      return render_template('login.html')

@app.route('/analysis')
def analysis():
      df = pd.read_csv(path)
      return render_template('analysis.html',destab=[(df.describe()).to_html(classes="destab")])

@app.route('/single')
def single():
      return render_template('single.html')

@app.route('/predict',methods=['POST'])
def predict():
      feature = [int(x) for x in request.form.values()]
      MP_SC = transform_data()
      train_MPSC = MP_SC.copy()
      train_MPSC['lead_times']=train_MPSC['lead_times'].astype(int)
      train_MPSC['shipping_times']= train_MPSC['shipping_times'].astype(int)
      train_MPSC['lead_time']= train_MPSC['lead_time'].astype(int)
      train_MPSC['manufacturing_lead_time']= train_MPSC['manufacturing_lead_time'].astype(int)
      X=train_MPSC[[ 'sku', 'price',
        'revenue_generated',
        'lead_times', 'shipping_times', 'shipping_costs',
       'lead_time', 'production_volumes', 'manufacturing_lead_time',
       'manufacturing_costs', 'inspection_results', 'defect_rates', 'routes', 'costs']]
      Y=train_MPSC[['stock_levels']]
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
      model1 = LinearRegression()
      model1.fit(X_train, Y_train)
    
      out = model1.predict(np.array([feature]))
      #first_name = request.form.get("sku")
      return render_template('prediction.html', x = round(out[0][0]))

@app.route('/predict1',methods=['POST'])
def predict1():
      feature = [int(x) for x in request.form.values()]
      MP_SC = transform_data()
      train_MPSC = MP_SC.copy()
      train_MPSC['lead_times']=train_MPSC['lead_times'].astype(int)
      train_MPSC['shipping_times']= train_MPSC['shipping_times'].astype(int)
      train_MPSC['lead_time']= train_MPSC['lead_time'].astype(int)
      train_MPSC['manufacturing_lead_time']= train_MPSC['manufacturing_lead_time'].astype(int)
      train_MPSC['Intercept'] = 1
      X=train_MPSC[['Intercept', 'sku', 'lead_times', 'order_quantities', 'location']]
      Y=train_MPSC[['costs']]
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
      model2 = LinearRegression()
      model2.fit(X_train, Y_train)
      feature.insert(0,1)
      out = model2.predict(np.array([feature]))
      return render_template('prediction.html', x1 = round(out[0][0]))


if __name__ == '__main__':

      #makeplot()
      



      app.run(debug=True)
