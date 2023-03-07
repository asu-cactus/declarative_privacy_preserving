from train import train_model, estimate_cost
from data_utils import (
    get_embeddings, 
    synthesize_database, 
    synthesize_simple_database,
    get_passenger_database,
    check_passenger_exist,
)
from test import evaluate
from global_variables import date_range, frequency_range
import numpy as np



def main(is_simple_data:bool = True):

    name_query = 'Alice Caine'
    datebirth = '1996-11-13'
    country = 'UK'
    picture_id = 2290 # we can link picture to its id.
    example_query = f"""
        SELECT img.location FROM virtual_surveillance_imgs img JOIN passengers ON match (passengers.pic, img) = True 
        WHERE passengers.name Like '\%{name_query}\%' AND datebirth='{datebirth}' AND country ='{country}'
    """
    print(f"Query is:\n{example_query}")
    
    embed_original, indices = get_embeddings() # Considered the query pictures
    if is_simple_data:
        embed_data, id_data, location_data = synthesize_simple_database(embed_original)
        date_data = None
    else:
        embed_data, id_data, date_data, location_data = synthesize_database(embed_original)

    passenger_data = get_passenger_database(embed_original, indices)
    query_image_index = check_passenger_exist(passenger_data, name_query, datebirth, country, picture_id)
    if query_image_index == -1:
        raise ValueError("Passenger doesn't exist in the database")
    else:
        print(f"query_image_index: {query_image_index}")
    
    # Get inputs to the model
    
    if query_image_index >= len(embed_original):
        raise ValueError("query image index exceed the total number of pictures") 
    print("Picture:") # TODO: show picture
    embed_query = embed_original[query_image_index]
    print(f"Embedding:\n {embed_query}")

    
    if is_simple_data:
        truth_label = location_data[query_image_index]
    else:
        query_date_index = int(input('Select date:')) # this is actually the index of the date for the same person
        if query_date_index >= frequency_range:
            raise ValueError("date exceed the toal number of dates") 
        label_index = query_image_index * frequency_range + query_date_index
        truth_label = location_data[label_index]

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
        if is_simple_data:
            location_pred = model.predict(np.expand_dims(embed_query, axis=0))
        else:
            date_encode = np.zeros(date_range, dtype=np.int32)
            input_ = np.concatenate((embed_query, date_encode), axis=0)
            location_pred = model.predict(np.expand_dims(input_, axis=0))
        location_pred = np.argmax(location_pred, axis = 1)


    location_pred += 1
    print(f'Predicted location is: {location_pred}')
    print(f'True location is: {truth_label}') # incorrect


if __name__ == '__main__':
    main()