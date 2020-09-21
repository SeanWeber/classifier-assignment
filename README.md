# Assignment classifier 

## Setup

Install Docker and Docker compose. Then run: 

    docker compose up -d

This will load the prediction API service on port 80.

## Model Training

The training script trains the model and saves the result to the models 
directory. A model is already saved, but if you wish to rebuild it, you can run

    docker build -t trainer .
    docker run -v [project directory]/models:/models trainer

For the scope of this project, versioning through git is enough.

## Prediction service

The prediction service takes instances and classifies them. It loads the 
trained model on startup.

### The `/classify` endpoint

This takes two query string parameters. Both are optional.

- `numeric0` - An integer
- `categorical0` - A string that matches either 'a', 'b', or 'c'

And returns

- `result` - Either a 0 or 1

You can see more information at the `/docs` endpoint.

## Discussion

### Model used

Various models were tried, including ensemble methods such as random forests.
However, they all performed equally as well. Given a choice between a simple
model and a complex model, the simple model is preferable.

### Variables used

A mutual information classifier was ran on the data. It was determined that 
only two variables affected the target, `numeric0` and `categorical0`. In order
to reduce network overhead, only the two relevant variables are sent.

| Feature | Mutual Information |
|---------|--------------------|
| 	categorical0_a |	0.0816113 |
|	numeric0       |	0.0473467 |
| 	categorical0_b |	0.0281643 |
| 	categorical0_c |	0.0259607 |
| 	day_of_month   |	<0.01 |
| 	month 	       | <0.01 |
| 	seconds 	   | <0.01 |
| 	numeric1       |	0 |
| 	timestamp      |	0 |
| 	hour           |	0 |
| 	year           | 	0 |
| 	day_of_week    |	0 |

It is also worth noting that `categorical0_b` and `categorical0_c` have no 
mutual information when rows where `categorical0 == a` are removed.
### Metric used

A binary classification problem such as this one is usually judged on one of
four metrics: accuracy, recall, precision, and F1 score. Which one to use
depends on the specific use case. For example if it was more important to find
true positives, we could bias the model to maximize recall at the expense of
precision. Since this was an exercise of arbitrary data, I just defaulted to 
accuracy.

### FastAPI

FastAPI was chosen for its performance, ease of use, and integration with the
rest of the Python ecosystem. Flask is another common alternative, however 
FastAPI offers lower latency, making it a better choice for real time 
applications.

### Docker

Docker is the industry standard for containerization.
