from train import train_model, estimate_cost
from data_utils import get_embeddings, synthesize_database
from test import evaluate
from global_variables import date_range
import numpy as np

def main():
    embed_original = get_embeddings() # Considered the query pictures
    embed_data, id_data, date_data, location_data = synthesize_database(embed_original)


    # Get inputs
    query_image_index = int(input('Select query image index:'))
    
    if query_image_index >= len(embed_original):
        raise ValueError("query image index exceed the total number of pictures") 
    print("Picture:") # TODO: show picture
    embed_query = embed_original[query_image_index]
    print(f"Embedding:\n {embed_query}")

    query_date = int(input('Select date:'))
    if query_date >= date_range:
        raise ValueError("date exceed the toal number of dates") 


    # Estimate cost
    costs = estimate_cost()
    print(f"Plan 1: eps: {costs[0]['eps']}, estimated accuracy: {costs[0]['acc']}")
    print(f"Plan 2: eps: {costs[1]['eps']}, estimated accuracy: {costs[1]['acc']}")

    # Select a plan and train model
    plan_selection = input("Select a plan:")
    if plan_selection == '1':
        model, eps, X_train, y_train = train_model(embed_data, id_data, date_data, location_data, user_selection='multi-output')
    else:
        model, eps, X_train, y_train = train_model(embed_data, id_data, date_data, location_data, user_selection='multi-input')

    # Evaluate model
    loss, acc = evaluate(model, X_train, y_train)
    print(f"eps: {eps[0]:.2f}, acc: {acc:.2f}, optimal RDP order: {eps[1]}, loss: {loss:.2f}")

    # Get prediction (location)
    if plan_selection == '1':
        location_pred = model.predict(np.expand_dims(embed_original[query_image_index], axis=0))
        location_pred = np.argmax(location_pred, axis = 1)
        location_pred %= location_pred     
    elif plan_selection == '2':
        date_encode = np.zeros(date_range, dtype=np.int32)
        input_ = np.concatenate((embed_query, date_encode), axis=0)
        location_pred = model.predict(np.expand_dims(input_, axis=0))
        location_pred = np.argmax(location_pred, axis = 1)
    location_pred += 1
    print(f'Predicted location is: {location_pred}')
    print(f'True location is: {location_data[query_image_index]}') # incorrect


if __name__ == '__main__':
    main()