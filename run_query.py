import pandas as pd
import numpy as np
import csv
import time

from data_utils import (
    get_embeddings, 
    synthesize_database, 
)

def create_image_dataset():
    embed_original, indices = get_embeddings()
    embed_data, id_data, date_data, location_data = synthesize_database(embed_original)
    list_columns = ["ImageID","Date","Location"] 

    with open('surveillence_img.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow (list_columns)
        for i in range(len(embed_original)):
            writer.writerow([id_data[i], date_data[i], location_data[i]])
        csvfile.close()
        

def query_location(name, datebirth, country):
    passenger = pd.read_csv("dataset/passenger_data/passenger.csv")
    image = pd.read_csv("dataset/surveillence_img.csv")
    embed_original, indices = get_embeddings()
    example_query = f"""
        SELECT img.location FROM surveillence_img img JOIN passengers ON np.allclose(embeddings(photoes[passengers.id]),embeddings(faces[img.id])) == True 
        WHERE passengers.name Like '\%{name_query}\%' AND datebirth=='{datebirth}' AND country =='{country}'
    """
    print(f"Query is:\n{example_query}")
    begin = time.time()
    img_id = passenger.query('name==@name and dob==@datebirth and country==@country').iloc[0]['id']
    #print(img_id)
    #print(len(embed_original))
    query_embedding = embed_original[img_id]
    #print (query_embedding)
    ret = -1
    for i in range (len(image)):
        if (np.allclose(embed_original[i], query_embedding)):
                ret= image.loc[i]["Location"]
                break;
    end = time.time()
    print("Elapsed time for query based on embedding matching:", end-begin, " seconds");
    return ret


def query_location_simple(name, datebirth, country):
    passenger = pd.read_csv("dataset/passenger_data/passenger.csv")
    image = pd.read_csv("dataset/surveillence_img.csv")
    example_query = f"""
        SELECT img.location FROM surveillence_img img JOIN passengers ON img.ImageID = passengers.ID
        WHERE passengers.name Like '\%{name_query}\%' AND datebirth=='{datebirth}' AND country =='{country}'
    """
    print(f"Query is:\n{example_query}")
    begin = time.time()
    location = pd.merge(image, passenger, left_on=['ImageID'], right_on=['id'], how='inner').query('name==@name and dob==@datebirth and country==@country').iloc[0]['Location']
    end = time.time()
    print("Elapsed time for query based on ID matching:", end-begin, " seconds");
    return location



if __name__ == '__main__':

    #create_image_dataset()

    # name_query = 'Alice Caine'
    # datebirth = '1996-11-13'
    # country = 'UK'
    # picture_id = 2290 # we can link picture to its id.

    name_query = 'Peter Derr'
    datebirth = '1982-06-05'
    country = 'UK'
    picture_id = 1099 # we can link picture to its id.
    location = query_location(name_query, datebirth, country)
    print("Location queried through embedding matching:", location)
    print("\n\n")
    location1 = query_location_simple(name_query, datebirth, country)
    print("Location queried through ID matching:", location1)

