
from time import time
import numpy as np

from data_utils import (
    get_embeddings,
    synthesize_simple_database,
    get_passenger_database,
    check_passenger_exist,
    gaussian_noise_to_embeddings,
    create_simple_trainset
)
from train import train_model, estimate_cost
from test import evaluate
from global_variables import (
    sigma,
    clip,
    delta,
    learning_rate,
    epochs,
    l2_norm_clip,
    noise_multiplier,
    batch_size,
    delta,
)


def run(
    learning_rate,
    epochs,
    l2_norm_clip,
    noise_multiplier,
    batch_size,
    delta,
    units
) -> tuple[float, float]:

    kwargs = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'l2_norm_clip': l2_norm_clip,
        'noise_multiplier': noise_multiplier,
        'batch_size': batch_size,
        'delta': delta,
        'units': units
    }
    latency = 0.0
    message = "Elapsed time for query based on privacy preserving is {} seconds"

    # Query
    name_query = 'Alice Caine'
    datebirth = '1996-11-13'
    country = 'UK'
    picture_id = 14  # we can link picture to its id.

    # name_query = 'Peter Derr'
    # datebirth = '1982-06-05'
    # country = 'UK'
    # picture_id = 18 # we can link picture to its id.

    example_query = f"""
        SELECT img.location FROM virtual_surveillance_imgs img JOIN passengers ON match (passengers.pic, img) = True 
        WHERE passengers.name Like '\%{name_query}\%' AND datebirth='{datebirth}' AND country ='{country}'
    """
    print(f"Example query:\n{example_query}")

    # Airport dataset
    embed_original, indices = get_embeddings()  # Considered the query pictures

    embed_data, id_data, location_data = synthesize_simple_database(
        embed_original)
    date_data = None

    embed_data = np.stack(embed_data)
    noisy_embed_data, epsilon = gaussian_noise_to_embeddings(
        embed_data, sigma, clip, delta)

    # Passenger database
    passenger_data = get_passenger_database(embed_original)
    begin = time()
    query_image_index = check_passenger_exist(
        passenger_data, name_query, datebirth, country, picture_id)
    latency += time() - begin
    if query_image_index == -1:
        print(message.format(latency))
        raise ValueError("Passenger doesn't exist in the database")
    else:
        print(f"query_image_index: {query_image_index}")

    # Get query embedding and its ground truth label
    if query_image_index >= len(embed_original):
        raise ValueError(
            "query image index exceed the total number of pictures")
    print("Picture:")  # Show picture
    embed_query = embed_original[query_image_index]
    print(f"Embedding:\n {embed_query}")

    truth_label = location_data[query_image_index]

    # Estimate cost
    costs = estimate_cost()

    print(
        f"Plan 1: eps: {costs[0]['eps']}, estimated accuracy: {costs[0]['acc']}")
    print(
        f"Plan 2: eps: {costs[1]['eps']}, estimated accuracy: {costs[1]['acc']}")

    # Select a plan and train model, hard coded
    plan_selection = '2'
    assert plan_selection == '1' or plan_selection == '2'
    embed_input = noisy_embed_data if plan_selection == '1' else embed_data
    is_privacy_preserve = False if plan_selection == '1' else True
    user_selection = 'single-output'

    model, eps, X_train, y_train, scaler = train_model(
        embed_input, id_data, date_data, location_data,
        user_selection=user_selection,
        is_privacy_preserve=is_privacy_preserve,
        **kwargs)

    # Evaluate model
    print('\n\nEvaluation:')

    if plan_selection == '1':
        X_eval, ids, y_train, scaler = create_simple_trainset(
            embed_data, id_data, location_data)
        loss, acc = evaluate(model, X_eval, y_train)
        print(
            f"[Evaluation] eps: {epsilon:.2f}, acc: {acc:.2f}, loss: {loss:.2f}")
    else:
        loss, acc = evaluate(model, X_train, y_train)
        print(
            f"[Evaluation] eps: {eps[0]:.2f}, acc: {acc:.2f}, optimal RDP order: {eps[1]}, loss: {loss:.2f}")

    # Get prediction (location), hard coded

    begin = time()
    location_pred = model.predict(scaler.transform(
        np.expand_dims(embed_query, axis=0)))
    location_pred = np.argmax(location_pred, axis=1)

    location_pred += 1
    latency += time() - begin
    print(message.format(latency))
    print(f'Predicted location is: {location_pred}')
    print(f'Ground truth location is: {truth_label}')
    return (epsilon, acc) if plan_selection == '1' else (eps[0], acc)


if __name__ == '__main__':
    run(
        learning_rate=0.02,
        epochs=70,
        l2_norm_clip=1,
        noise_multiplier=0.3,
        batch_size=200,
        delta=1e-5,
        units=500)
