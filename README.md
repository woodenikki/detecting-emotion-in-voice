## Business Understanding

- **Stakeholders**: apple growers, distributors, and retailers
- **Problem**: 
> By predicting apple quality, this project helps the entire apple industry. Growers can prioritize harvesting the best apples, reducing waste. Distributors can send high-quality apples to premium markets, ensuring customer satisfaction. Retailers can optimize inventory, reducing spoilage and improving profitability. Overall, this project helps reduce waste and improve customer satisfaction for all stakeholders.

---

## Data Understanding:
This dataset contains information about various attributes of a set of fruits, providing insights into their characteristics. The dataset includes details such as fruit ID, size, weight, sweetness, crunchiness, juiciness, ripeness, acidity, and quality.

The dataset was generously provided by an American agriculture company. The data has been scaled and cleaned for ease of use.

[Apple Quality: Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality)

#### Features:

| Feature     | Description                                                |
|-------------|-----------------------------------------------------------|
| A_id        | Unique identifier for each fruit                         |
| Size        | Size of the fruit                                        |
| Weight      | Weight of the fruit                                      |
| Sweetness   | Degree of sweetness of the fruit                         |
| Crunchiness | Texture indicating the crunchiness of the fruit          |
| Juiciness   | Level of juiciness of the fruit                          |
| Ripeness    | Stage of ripeness of the fruit                           |
| Acidity     | Acidity level of the fruit                               |
| Quality     | Overall quality of the fruit (target variable)           |

### Data Limitations
- **Limited sample size** - The dataset only contains 4,000 samples, which may not capture the full range of variability in apple quality across different varieties, growing conditions, and regions. 
- **Lack of contextual information** - The dataset gives us features like size, weight, sweetness, crunchiness, juiciness, ripeness, and acidity, but we were not given any context about the varieties of apples, or how these features were extracted. This is important context.
- **Binary quality labels** - We have binary labels for the 'Quality' table: 'good' and 'bad'. This simplification is helpful for creating a model, but applie quality is likely more nuanced, and would benefit from a more granular labeling system.
- **Normalized features** - this dataset's features have already been normalized, which is helpful for creating a model, but might make it harder to relate the model's predictions to real-world measurements. 

---

## Data Preperation
It looks like the features in our database have been standardized and normalized -- (likely because it's difficult to create a scale for subjective 'scores' such as sweetness and juiciness) this is very helpful.

Let's continue with the rest of our data preparation steps:

- Handling Missing Values
- Encoding Categorical Variables
- Feature Scaling
- Feature Selection / Engineering

---

## Modeling

We created multiple models, increasing in complexity. The final model (with best performance metrics) was built using LightGBM, a powerful and efficient gradient boosting framework for Python, to implement our ensemble model.
> [LightGBM Docs](https://lightgbm.readthedocs.io/en/stable/)

### Gradient Boosting Model

LightGBM Accuracy: 0.9

LightGBM Classification Report:

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.90      | 0.90   | 0.90     | 401     |
| 1            | 0.90      | 0.90   | 0.90     | 399     |
| accuracy     |           |        | 0.90     | 800     |
| macro avg    | 0.90      | 0.90   | 0.90     | 800     |
| weighted avg | 0.90      | 0.90   | 0.90     | 800     |


#### Comparison
Decision Tree Classifier Accuracy: 0.81
LightGBM Accuracy: 0.9

The LightGBM model has achieved an accuracy of 0.90, which meets our desired accuracy score of at least 90%. The precision, recall, and f1-score for both classes (0 and 1) are also 0.90, indicating strong performance on all metrics. 

---

## Evaluation of Final Model

#### Final Model: Gradient Boosting
The LightGBM Model stands out as the best performer on all metrics. 

#### Justification
For this classification problem, precision, recall, and f1-score are crucial metrics (because both false-positives and false-negatives would have significant real-world consequences)
- **precision** tells us how many apples predicted to be of good quality and actually were: **90%**
- **recall** tells us how many of the truly good apples were correctly identified: **90%**
- **f1-score** represents the balance between precision and recall: **90%**

This final model is a reliable tool for automating apple quality assessment. Including:
- **Operational Efficiency**: this model can save time and resources, reducing the need for manual inspection. 
- **Minimize Waste**: Accurate classification reduces the risk of good apples being thrown out, or bad ones mistakenly being included in shipments.
- **Customer Satisfaction**: ensuring that only high-quality apples reach the market will encrease brand repuation and customer loyalty. 

#### Recommendations:
- Include the LightGBM Model in quality assurance (QA) process to improve efficiency
- Keep human inspection in the QA process. While this model is highly accurate, it may still miss subtle or edge cases. 
- Periodically retrain the model to adapt to changing factors (seasonal factors, new apple varieties)
- Implement monitoring to track performance in real-world scenarios to ensure consistency. 