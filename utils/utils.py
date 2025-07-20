import numpy as np
import imageio
import os
from skimage.morphology import label

from parameter import *


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    """
    Converts world coordinates to cell positions based on map information.

    Parameters
    ----------
    coords : np.ndarray
        Array of coordinates to convert. Can be of shape (2,) for a single coordinate or (N, 2) for multiple coordinates.
    map_info : object
        An object containing map metadata, specifically `map_origin_x`, `map_origin_y`, and `cell_size` attributes.
    check_negative : bool, optional
        If True, asserts that all resulting cell positions are non-negative.

    Returns
    -------
    np.ndarray or tuple
        The cell position(s) as integer coordinates. Returns a single tuple for one coordinate, or an array of tuples for multiple coordinates.

    Raises
    ------
    AssertionError
        If `check_negative` is True and any resulting cell position is negative.

    Notes
    -----
    - The function rounds the calculated cell positions to the nearest integer.
    - If a single coordinate is provided, returns a single cell position; otherwise, returns an array of cell positions.
    """
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords, map_info.map_origin_x, map_info.map_origin_y)
    if single_cell:
        return cell_position[0]
    else:
        return cell_position

def get_coords_from_cell_position(cell_position, map_info):
    """
    Converts cell positions to map coordinates based on map information.

    Parameters:
        cell_position (np.ndarray): Array of cell positions with shape (N, 2), where each row is (x, y).
        map_info (object): An object containing map metadata with attributes:
            - cell_size (float): Size of each cell in map units.
            - map_origin_x (float): X-coordinate of the map origin.
            - map_origin_y (float): Y-coordinate of the map origin.

    Returns:
        np.ndarray: Array of map coordinates corresponding to the input cell positions, rounded to one decimal place.
                    If the number of coordinates equals OCCUPIED, returns the first coordinate only.

    Notes:
        - The function expects cell_position to be convertible to shape (-1, 2).
        - OCCUPIED should be defined elsewhere in the codebase.
    """
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == OCCUPIED:
        return coords[0]
    else:
        return coords

def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    """
    Returns a boolean map indicating the free and connected area in the map starting from a given location.

    Args:
        location (tuple or list): The coordinates (x, y) representing the starting location.
        map_info (object): An object containing the map data and relevant attributes. Must have a 'map' attribute.

    Returns:
        numpy.ndarray: A boolean array where True values indicate free and connected cells accessible from the given location.

    Notes:
        - The function assumes the existence of constants and functions such as FREE, label, and get_cell_position_from_coords.
        - Connectivity is determined using 8-connectivity (connectivity=2).
    """
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map

def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    """
    Generates a list of node coordinates within a map region that are either free or free and connected, 
    depending on the connectivity check.

    Args:
        location (tuple or array-like): The reference location used for connectivity checking.
        updating_map_info (object): An object containing map information, including origin coordinates, 
            map shape, and the map array itself.
        check_connectivity (bool, optional): If True, only nodes that are both free and connected are returned.
            If False, only free nodes are returned. Defaults to True.

    Returns:
        nodes (np.ndarray): Array of shape (N, 2) containing the coordinates of the selected nodes.
        free_connected_map (np.ndarray or None): Array representing the free and connected map if 
            check_connectivity is True, otherwise None.

    Notes:
        - The function aligns the map boundaries to the node resolution.
        - Node coordinates are rounded to one decimal place.
        - Requires global constants: CELL_SIZE, NODE_RESOLUTION, FREE.
        - Depends on helper functions: get_cell_position_from_coords, get_free_and_connected_map.
    """
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map

def get_frontier_in_map(map_info):
    '''
    Get frontier cells in the map.
    '''
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
     # create a binary map where UNKNOWN cells are 1 and others are 0
    unknown = (map_info.map == UNKNOWN) * 1
    # Pad the unknown map by 1 cell on each side to prevent out-of-bounds errors
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)

    # Calculate the number of unknown neighbors for each interior cell of the map
    # The sum of these neighbors gives the number of unknown neighbors for each cell
    # The result is a matrix where each cell contains the count of unknown neighbors
    # unknown[2:][:, 1:x_len + 1] checks the cell below (chain indexing)
    # unknown[:y_len][:, 1:x_len + 1] checks the cell on top (note y_len is the pre-padded height i.e. unknown.shape[0] - 2)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    # frontier cells defined as cells with 2 to 7 unknown neighbors (inclusive)
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1 ,2)
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = set(map(tuple, frontier_coords))
    return frontier_coords

def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data

def is_frontier(location, map_info):
    cell = get_cell_position_from_coords(location, map_info)
    if map_info.map[cell[1], cell[0]] != FREE:
        return False
    else:
        assert cell[1] - 1 > 0 and cell[1] - 1 > 0 and cell[1] + 2 < map_info.map.shape[1] and cell[0] + 2 < map_info.map.shape[0]
        unknwon = map_info.map[cell[1] - 1:cell[1] + 2, cell[0] - 1: cell[0] + 2] == UNKNOWN
        n = np.sum(unknwon)
        if 1 < n < 8:
            return True
        else:
            return False

def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    for filename in frame_files[:-1]:
        os.remove(filename)

def make_gif_test(path, n, frame_files, rate, n_agents, fov, sensor_range):
    with imageio.get_writer('{}/{}_{}_{}_{}_explored_rate_{:.4g}.gif'.format(path, n, n_agents, fov, sensor_range, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')
    for filename in frame_files[:-1]:
        os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y


