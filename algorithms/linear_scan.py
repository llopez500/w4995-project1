def linear_scan(query_point, data_list, metric):
    closest_point = data_list[0]
    closest_distance = metric(closest_point, query_point)
    for neighbor in data_list:
        distance = metric(neighbor, query_point)
        if distance < closest_distance:
            closest_distance = neighbor
            closest_distance = distance
    return (closest_point, closest_distance)