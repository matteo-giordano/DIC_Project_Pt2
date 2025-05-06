#!/usr/bin/env python3
"""
Flask web application for building maze datasets.
"""

import os
import flask
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pathlib import Path
import numpy as np
import matplotlib
# Set non-interactive backend before importing pyplot to avoid thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import json
import threading
import yaml
from typing import Dict, List, Tuple, Any, Optional
from grid_generator import (
    PrimMazeGenerator,
    RecursiveDivisionMazeGenerator,
    WilsonMazeGenerator,
    TerrainGenerator,
    GaussianNoiseGenerator,
    DatasetGenerator
)
from dataset import Dataset

# Custom Manual Drawing Generator class
class ManualDrawingGenerator:
    def __init__(self, width, height, maze_data=None):
        """
        Initialize the manual drawing generator.
        
        Args:
            width (int): Width of the maze
            height (int): Height of the maze
            maze_data (np.ndarray, optional): Predefined maze data
        """
        self.width = width
        self.height = height
        self.maze_data = maze_data
    
    def generate_map(self) -> np.ndarray:
        """
        Return the predefined maze data.
        
        Returns:
            np.ndarray: The manually drawn maze
        """
        if self.maze_data is None:
            # If no maze data, create an empty maze with borders
            maze = np.zeros((self.height, self.width), dtype=np.int8)
            maze[0, :] = maze[-1, :] = 1  # Top and bottom borders
            maze[:, 0] = maze[:, -1] = 1  # Left and right borders
            return maze
        
        return self.maze_data

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session management

# Lock to prevent concurrent generation requests
generation_lock = threading.Lock()

# Global variables for dataset exploration
MAZES_PER_PAGE = 16

# Generator info with default parameters
GENERATORS = {
    'prim': {
        'name': 'Prim Algorithm',
        'class': PrimMazeGenerator,
        'params': {}
    },
    'recursive': {
        'name': 'Recursive Division',
        'class': RecursiveDivisionMazeGenerator,
        'params': {}
    },
    'wilson': {
        'name': 'Wilson\'s Algorithm',
        'class': WilsonMazeGenerator,
        'params': {}
    },
    'terrain': {
        'name': 'Terrain Generator',
        'class': TerrainGenerator,
        'params': {
            'blob_density': {
                'type': 'number',
                'default': 0.005,
                'min': 0.001,
                'max': 0.02,
                'step': 0.001,
                'label': 'Blob Density'
            }
        }
    },
    'gaussian': {
        'name': 'Gaussian Noise',
        'class': GaussianNoiseGenerator,
        'params': {
            'threshold': {
                'type': 'number',
                'default': 0.65,
                'min': 0.5,
                'max': 0.8,
                'step': 0.01,
                'label': 'Threshold'
            },
            'sigma_x': {
                'type': 'number',
                'default': 1.0,
                'min': 0.1,
                'max': 3.0,
                'step': 0.1,
                'label': 'Sigma X'
            },
            'sigma_y': {
                'type': 'number',
                'default': 1.0,
                'min': 0.1,
                'max': 3.0,
                'step': 0.1,
                'label': 'Sigma Y'
            }
        }
    },
    'manual': {
        'name': 'Manual Drawing',
        'class': ManualDrawingGenerator,
        'params': {
            'grid_size': {
                'type': 'number',
                'default': 21,
                'min': 9,
                'max': 51,
                'step': 2,
                'label': 'Grid Size'
            }
        }
    }
}

# Common maze size options
MAZE_SIZES = [
    {'width': 11, 'height': 11, 'name': '11×11'},
    {'width': 21, 'height': 21, 'name': '21×21'},
    {'width': 31, 'height': 31, 'name': '31×31'},
    {'width': 41, 'height': 41, 'name': '41×41'},
    {'width': 51, 'height': 51, 'name': '51×51'}
]

def load_generator_config():
    """
    Load generator configurations from dataset_config.yaml file.
    
    Returns:
        dict: Dictionary containing generator configurations from the YAML file
    """
    config_path = Path(__file__).parent / 'dataset_config.yaml'
    default_config = {
        'name': 'default_dataset',
        'generators': {
            'prim': {
                'count_per_config': 10,
                'width': [21],
                'height': [21]
            },
            'recursive': {
                'count_per_config': 10,
                'width': [21],
                'height': [21]
            },
            'wilson': {
                'count_per_config': 10,
                'width': [21],
                'height': [21]
            },
            'terrain': {
                'count_per_config': 10,
                'width': [21],
                'height': [21],
                'blob_density': [0.005]
            },
            'gaussian': {
                'count_per_config': 10,
                'width': [21],
                'height': [21],
                'threshold': [0.65],
                'sigma_x': [1.0],
                'sigma_y': [1.0]
            }
        }
    }
    
    try:
        if not config_path.exists():
            return default_config
            
        with open(config_path, 'r') as f:
            yaml_content = f.read()
            config = yaml.safe_load(yaml_content)
            
            if not config:
                return default_config
                
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config

@app.route('/')
def index():
    """Render the home page."""
    session.clear()  # Clear any existing session data
    
    # Set default dataset name
    if 'dataset_name' not in session:
        session['dataset_name'] = f"maze_dataset_{datetime.now().strftime('%Y%m%d')}"
    
    return render_template('dataset_builder/index.html', 
                          generators=GENERATORS, 
                          dataset_name=session.get('dataset_name', ''))

@app.route('/create_dataset')
def create_dataset():
    """Show the create dataset page."""
    if 'dataset_name' not in session:
        session['dataset_name'] = f"maze_dataset_{datetime.now().strftime('%Y%m%d')}"
    
    # Get generator type from URL parameters or default to 'prim'
    generator_type = request.args.get('generator', 'prim')
    
    # Ensure it's a valid generator
    if generator_type not in GENERATORS:
        generator_type = 'prim'
    
    # Store the current generator type in session
    session['current_generator'] = generator_type
    
    # Get generator info
    generator_info = GENERATORS[generator_type]
    
    # Load generator configuration from YAML
    generator_config = load_generator_config()
    
    # For manual drawing generator, use a different template
    if generator_type == 'manual':
        return render_template('dataset_builder/manual_draw.html',
                              generators=GENERATORS,
                              current_generator=generator_type,
                              generator_info=generator_info,
                              dataset_name=session.get('dataset_name', ''),
                              generator_config=generator_config)
    
    return render_template('dataset_builder/create.html',
                          generators=GENERATORS,
                          current_generator=generator_type,
                          generator_info=generator_info,
                          maze_sizes=MAZE_SIZES,
                          dataset_name=session.get('dataset_name', ''),
                          generator_config=generator_config)

@app.route('/update_dataset_name', methods=['POST'])
def update_dataset_name():
    """Update the dataset name."""
    data = request.get_json()
    new_name = data.get('name', '')
    
    if new_name:
        session['dataset_name'] = new_name
        return jsonify({'success': True, 'name': new_name})
    else:
        return jsonify({'success': False, 'error': 'Invalid name'})

@app.route('/generate_maps', methods=['POST'])
def generate_maps():
    """Generate maps with the specified parameters."""
    # Check if a generation is already in progress
    if not generation_lock.acquire(blocking=False):
        return jsonify({
            'success': False,
            'error': 'Map generation already in progress. Please wait for the current generation to complete.'
        })
    
    try:
        generator_type = request.form.get('generator_type')
        width = int(request.form.get('width', 21))
        height = int(request.form.get('height', 21))
        num_samples = int(request.form.get('num_samples', 8))
        ensure_connected = request.form.get('ensure_connected') == 'true'
        ensure_unique = request.form.get('ensure_unique') == 'true'
        
        # Get generator-specific parameters
        generator_params = {}
        generator_info = GENERATORS.get(generator_type, {})
        
        for param_name, param_info in generator_info.get('params', {}).items():
            param_type = param_info['type']
            param_value = request.form.get(param_name, param_info['default'])
            
            if param_type == 'number':
                # Handle floating point numbers
                if '.' in str(param_value) or '.' in str(param_info['step']):
                    generator_params[param_name] = float(param_value)
                else:
                    generator_params[param_name] = int(param_value)
            else:
                generator_params[param_name] = param_value
        
        # Generate the maps
        generator_class = generator_info.get('class')
        if not generator_class:
            return jsonify({'success': False, 'error': 'Invalid generator type'})
        
        # Limit the number of samples to a reasonable number
        num_samples = min(num_samples, 16)
        maps = []
        
        # Information about deduplication
        deduplication_info = {
            'enabled': ensure_unique,
            'total_generated': 0,
            'duplicates_removed': 0
        }
        
        # Generate initial batch of mazes
        for _ in range(num_samples):
            generator = generator_class(width, height, **generator_params)
            maze = generator.generate_map()
            deduplication_info['total_generated'] += 1
            
            # If requested, ensure connectivity
            if ensure_connected:
                # Create a temporary dataset to use the ensure_connectivity method
                temp_dataset = Dataset()
                temp_dataset.maps = [maze]
                temp_metadata = {
                    'generator_type': generator_type,
                    'width': width, 
                    'height': height
                }
                temp_dataset.metadata = [temp_metadata]
                
                obstacles_removed = temp_dataset.ensure_connectivity(0)
                maze = temp_dataset.maps[0]
            
            maps.append(maze)
        
        # If uniqueness is requested and we have multiple maps, deduplicate them
        if ensure_unique and len(maps) > 1:
            # Create a temporary DatasetGenerator to use the deduplication method
            temp_generator = DatasetGenerator()
            temp_generator.dataset = maps
            
            # Create metadata for each map
            temp_generator.metadata = []
            for i in range(len(maps)):
                temp_generator.metadata.append({
                    'generator_type': generator_type,
                    'width': width,
                    'height': height
                })
            
            # Run deduplication
            removed_count = temp_generator.deduplicate_dataset()
            deduplication_info['duplicates_removed'] += removed_count
            
            max_attempts = 50  # Prevent infinite loops
            attempts = 0
            
            # If deduplication removed maps, generate additional ones to maintain the requested count
            while len(temp_generator.dataset) < num_samples and attempts < max_attempts:
                attempts += 1
                # Generate more mazes than we need to increase chances of finding unique ones
                additional_needed = num_samples - len(temp_generator.dataset)
                batch_size = min(additional_needed * 2, 10)  # Generate up to 10 at a time
                
                new_mazes = []
                for _ in range(batch_size):
                    generator = generator_class(width, height, **generator_params)
                    new_maze = generator.generate_map()
                    deduplication_info['total_generated'] += 1
                    
                    # Ensure connectivity if requested
                    if ensure_connected:
                        temp_dataset = Dataset()
                        temp_dataset.maps = [new_maze]
                        temp_metadata = {'generator_type': generator_type, 'width': width, 'height': height}
                        temp_dataset.metadata = [temp_metadata]
                        
                        temp_dataset.ensure_connectivity(0)
                        new_maze = temp_dataset.maps[0]
                    
                    new_mazes.append(new_maze)
                
                # Add new mazes to the dataset
                original_count = len(temp_generator.dataset)
                for maze in new_mazes:
                    temp_generator.dataset.append(maze)
                    temp_generator.metadata.append({
                        'generator_type': generator_type,
                        'width': width,
                        'height': height
                    })
                
                # Run deduplication again
                prev_duplicates_removed = deduplication_info['duplicates_removed']
                removed_count = temp_generator.deduplicate_dataset()
                deduplication_info['duplicates_removed'] += removed_count
                
                # If we can't generate any more unique mazes after multiple attempts, break
                if original_count == len(temp_generator.dataset) and attempts > 5:
                    break
            
            # Use the deduplicated maps
            maps = temp_generator.dataset[:num_samples]
            
            # Update deduplication info
            deduplication_info['final_unique_count'] = len(maps)
            if len(maps) < num_samples:
                deduplication_info['note'] = f"Could only generate {len(maps)} unique mazes with current parameters."
        
        # Convert maps to base64 images
        map_images = []
        for i, maze in enumerate(maps):
            map_images.append({
                'image': maze_to_base64(maze),
                'index': i + 1,
                'info': f"{generator_info.get('name')} {width}×{height}"
            })
        
        # Add deduplication info to the response if enabled
        response_data = {
            'success': True,
            'maps': map_images,
            'info': {
                'generator': generator_info.get('name'),
                'width': width,
                'height': height,
                'count': len(map_images)
            }
        }
        
        if ensure_unique:
            response_data['deduplication_info'] = deduplication_info
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        # Always release the lock when done
        generation_lock.release()

def maze_to_base64(maze):
    """Convert a maze to a base64 encoded image."""
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create a colored grid
    colored_maze = np.zeros((*maze.shape, 3))
    colored_maze[maze == 0] = [1, 1, 1]  # White for empty spaces
    colored_maze[maze == 1] = [0, 0, 0]  # Black for walls
    
    ax.imshow(colored_maze)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # Hide axes
    
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/save_config', methods=['POST'])
def save_config():
    """Save the configuration to dataset_config.yaml."""
    try:
        data = request.get_json()
        yaml_content = data.get('config', '')
        ensure_connected = data.get('ensure_connected', True)
        ensure_unique = data.get('ensure_unique', True)
        
        if not yaml_content:
            return jsonify({'success': False, 'error': 'Empty configuration'})
        
        # Validate YAML syntax
        try:
            # If yaml_content is already a dict, use it directly
            if isinstance(yaml_content, dict):
                config = yaml_content
            else:
                # Otherwise, load it as YAML string
                config = yaml.safe_load(yaml_content)
                
            if not config:
                return jsonify({'success': False, 'error': 'Invalid YAML structure'})
                
            # Basic validation of expected structure
            if 'name' not in config:
                return jsonify({'success': False, 'error': 'Missing dataset name in configuration'})
                
            if 'generators' not in config:
                return jsonify({'success': False, 'error': 'Missing generators section in configuration'})
            
            # Add ensure_connected and ensure_unique options
            config['ensure_connected'] = ensure_connected
            config['ensure_unique'] = ensure_unique
            
            # Convert to YAML string
            yaml_content = yaml.dump(config, default_flow_style=False)
                
        except yaml.YAMLError as e:
            return jsonify({'success': False, 'error': f'YAML parsing error: {str(e)}'})
        
        # Get the path to the config file
        config_path = Path(__file__).parent / 'dataset_config.yaml'
        
        # Write the configuration to file
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_config', methods=['GET'])
def load_config():
    """Load the configuration from dataset_config.yaml."""
    try:
        config = load_generator_config()
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    """Load a dataset from the specified path and redirect to the explorer page."""
    dataset_path = request.form.get('dataset_path', '')
    
    if not dataset_path:
        return redirect(url_for('index'))
    
    try:
        # Convert path to a Path object
        dataset_path_obj = Path(dataset_path)
        
        # Check if the path exists
        if not dataset_path_obj.exists():
            # Try to resolve relative to the world directory
            world_dir = Path(__file__).parent
            alternative_path = world_dir / dataset_path_obj.name
            
            if alternative_path.exists():
                dataset_path_obj = alternative_path
            else:
                # If both paths don't exist, check if any path with a similar prefix exists
                prefix = dataset_path_obj.name.split('_')[0]  # Get the dataset name part
                potential_paths = list(world_dir.glob(f"{prefix}_*"))
                
                if potential_paths:
                    # Use the most recent dataset
                    dataset_path_obj = max(potential_paths, key=lambda p: p.stat().st_mtime)
                else:
                    # No suitable path found
                    error_message = f"Dataset path not found: {dataset_path}"
            return render_template('dataset_builder/index.html', 
                                generators=GENERATORS, 
                                dataset_name=session.get('dataset_name', ''),
                                        error=error_message)
        
        # Load the dataset
        dataset = Dataset()
        
        try:
            dataset.load(str(dataset_path_obj))
        except Exception as e:
            error_message = f"Error loading dataset from {dataset_path_obj}: {str(e)}"
            return render_template('dataset_builder/index.html', 
                                 generators=GENERATORS, 
                                 dataset_name=session.get('dataset_name', ''),
                                 error=error_message)
        
        # Store dataset info in session
        session['dataset_path'] = str(dataset_path_obj)
        session['dataset_name'] = dataset.dataset_name
        session['total_mazes'] = len(dataset.metadata)
        
        # Get generator types and dimensions for filters
        session['generator_types'] = dataset.get_generator_types()
        dimensions = dataset.get_dimensions()
        session['dimensions'] = [[dim[0], dim[1]] for dim in dimensions]  # Convert tuples to lists for JSON serialization
        
        # Initialize filters (all selected by default)
        session['selected_generators'] = session['generator_types']
        session['selected_dimensions'] = session['dimensions']
        
        # Store filter criteria
        session['filter_criteria'] = {}
        
        # Set current page
        session['current_page'] = 1
        
        return redirect(url_for('view_page', page=1))
        
    except Exception as e:
        error_message = f"Error loading dataset: {str(e)}"
        return render_template('dataset_builder/index.html', 
                             generators=GENERATORS, 
                             dataset_name=session.get('dataset_name', ''),
                             error=error_message)

@app.route('/apply_filters', methods=['POST'])
def apply_filters():
    """Apply filters to the dataset."""
    if 'dataset_path' not in session:
        return redirect(url_for('index'))
    
    # Get selected generator types
    selected_generators = request.form.getlist('generator_types')
    if not selected_generators:  # If none selected, use all
        selected_generators = session['generator_types']
    
    # Get selected dimensions
    selected_dimensions_raw = request.form.getlist('dimensions')
    selected_dimensions = []
    for dim_str in selected_dimensions_raw:
        try:
            width, height = map(int, dim_str.split(','))
            selected_dimensions.append([width, height])
        except ValueError:
            continue
    
    if not selected_dimensions:  # If none selected, use all
        selected_dimensions = session['dimensions']
    
    # Update session
    session['selected_generators'] = selected_generators
    session['selected_dimensions'] = selected_dimensions
    
    # Reset to first page
    session['current_page'] = 1
    
    return redirect(url_for('view_page', page=1))

@app.route('/view/<int:page>')
def view_page(page):
    """View a specific page of the dataset."""
    if 'dataset_path' not in session:
        return redirect(url_for('index'))
    
    # Ensure page is valid
    page = max(1, page)
    
    try:
        # Load dataset
        dataset = Dataset()
        dataset.load(session['dataset_path'])
        
        # Apply filters
        filtered_dataset = dataset
        filter_criteria = {}
        
        # Filter by generator type if needed
        if session['selected_generators'] != session['generator_types']:
            if len(session['selected_generators']) == 1:
                # If only one generator type is selected, we can use the more efficient generator_type parameter
                filter_criteria['generator_type'] = session['selected_generators'][0]
            else:
                # Otherwise, we need to build a more complex filter
                filter_criteria_list = []
                for gen_type in session['selected_generators']:
                    filter_criteria_list.append({'generator_type': gen_type})
        
        # Filter by dimensions
        if session['selected_dimensions'] != session['dimensions']:
            # For multiple dimensions, we need to construct a more complex filter
            if len(session['selected_dimensions']) > 1:
                # Build filter criteria for each selected dimension
                selected_filters = []
                
                # First, handle the case where we've already specified a generator type
                if 'generator_type' in filter_criteria:
                    gen_type = filter_criteria['generator_type']
                    # Build dimension filters for this generator type
                    for dim in session['selected_dimensions']:
                        dimension_filter = {
                            'generator_type': gen_type,
                            'width': dim[0],
                            'height': dim[1]
                        }
                        selected_filters.append(dimension_filter)
                else:
                    # No generator type specified yet, just filter by dimensions
                    for dim in session['selected_dimensions']:
                        dimension_filter = {
                            'width': dim[0],
                            'height': dim[1]
                        }
                        selected_filters.append(dimension_filter)
                
                # Apply the filters sequentially
                filtered_maps = []
                filtered_metadata = []
                
                for filter_params in selected_filters:
                    temp_dataset = dataset.filter(filter_params)
                    # Collect maps and metadata from this filtered dataset
                    for i in range(len(temp_dataset.maps)):
                        if temp_dataset.maps[i] is not None:
                            filtered_maps.append(temp_dataset.maps[i])
                            filtered_metadata.append(temp_dataset.metadata[i])
                
                # Create a new dataset with the filtered maps
                filtered_dataset = Dataset(dataset.dataset_name)
                filtered_dataset.maps = filtered_maps
                filtered_dataset.metadata = filtered_metadata
            else:
                # If only one dimension is selected, use simple filtering
                width, height = session['selected_dimensions'][0]
                filter_criteria['width'] = width
                filter_criteria['height'] = height
                
                # Apply filters
                filtered_dataset = dataset.filter(filter_criteria)
        
        # Apply filters if any (for cases not handled by dimension filtering)
        if filter_criteria and session['selected_dimensions'] == session['dimensions']:
            filtered_dataset = dataset.filter(filter_criteria)
        
        # Calculate pagination
        total_mazes = len(filtered_dataset.metadata)
        total_pages = (total_mazes + MAZES_PER_PAGE - 1) // MAZES_PER_PAGE
        
        # Adjust page if out of bounds
        page = min(page, total_pages) if total_pages > 0 else 1
        
        # Calculate start and end indices
        start_idx = (page - 1) * MAZES_PER_PAGE
        end_idx = min(start_idx + MAZES_PER_PAGE, total_mazes)
        
        # Get mazes for current page
        mazes = []
        maze_metadata = []
        maze_indices = []
        
        for idx in range(start_idx, end_idx):
            maze = filtered_dataset.get_map(idx)
            metadata = filtered_dataset.get_metadata(idx)
            
            if maze is not None and metadata is not None:
                mazes.append(maze.tolist())  # Convert to list for JSON serialization
                maze_metadata.append(metadata)
                maze_indices.append(idx)
        
        # Set page navigation
        prev_page = page - 1 if page > 1 else None
        next_page = page + 1 if page < total_pages else None
        
        # Update current page in session
        session['current_page'] = page
        
        # Store the filtered dataset in session for use in maze_detail
        session['filtered_indices'] = maze_indices
        
        return render_template('dataset_builder/explorer.html',
                              dataset_loaded=True,
                              dataset_name=session['dataset_name'],
                              total_mazes=total_mazes,
                              current_page=page,
                              total_pages=total_pages,
                              prev_page=prev_page,
                              next_page=next_page,
                              mazes=mazes,
                              maze_metadata=maze_metadata,
                              maze_indices=maze_indices,
                              generator_types=session['generator_types'],
                              dimensions=session['dimensions'],
                              selected_generators=session['selected_generators'],
                              selected_dimensions=session['selected_dimensions'])
    
    except Exception as e:
        # Handle any errors that occur during dataset loading or processing
        error_message = f"Error processing dataset: {str(e)}"
        return render_template('dataset_builder/index.html', 
                             generators=GENERATORS, 
                             dataset_name=session.get('dataset_name', ''),
                             error=error_message)

@app.route('/maze/<int:index>')
def maze_detail(index):
    """View detailed information about a specific maze."""
    if 'dataset_path' not in session:
        return redirect(url_for('index'))
    
    try:
        # Load dataset
        dataset = Dataset()
        dataset.load(session['dataset_path'])
        
        # Get the maze and its metadata
        maze = dataset.get_map(index)
        metadata = dataset.get_metadata(index)
        
        if maze is None or metadata is None:
            return redirect(url_for('view_page', page=session.get('current_page', 1)))
        
        # Generate a base64 image of the maze
        maze_img = maze_to_base64(maze)
        
        # Get the indices of filtered mazes
        filtered_indices = session.get('filtered_indices', [])
        
        # Find the current position in the filtered list
        try:
            current_pos = filtered_indices.index(index)
            total_filtered = len(filtered_indices)
            
            # Determine prev and next indices in the filtered set
            prev_index = filtered_indices[current_pos - 1] if current_pos > 0 else None
            next_index = filtered_indices[current_pos + 1] if current_pos < total_filtered - 1 else None
        except (ValueError, IndexError):
            # If the index is not in the filtered list, just use direct navigation
            prev_index = index - 1 if index > 0 else None
            next_index = index + 1 if index < session.get('total_mazes', 0) - 1 else None
            current_pos = index
            total_filtered = session.get('total_mazes', 0)
        
        return render_template('dataset_builder/maze_detail.html',
                             maze=maze.tolist(),
                             metadata=metadata,
                             maze_img=maze_img,
                             index=index,
                             prev_index=prev_index,
                             next_index=next_index,
                             current_pos=current_pos + 1,
                             total=total_filtered)
    
    except Exception as e:
        error_message = f"Error viewing maze detail: {str(e)}"
        return render_template('dataset_builder/index.html', 
                             generators=GENERATORS, 
                             dataset_name=session.get('dataset_name', ''),
                             error=error_message)

@app.route('/download_metadata/<int:index>')
def download_metadata(index):
    """Download the metadata for a specific maze as JSON."""
    if 'dataset_path' not in session:
        return redirect(url_for('index'))
    
    try:
        # Load dataset
        dataset = Dataset()
        dataset.load(session['dataset_path'])
        
        # Get the maze metadata
        metadata = dataset.get_metadata(index)
        
        if metadata is None:
            return jsonify({"error": "Metadata not found"}), 404
        
        return jsonify(metadata)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_full_dataset', methods=['POST'])
def generate_full_dataset():
    """Generate a full dataset based on the current configuration."""
    try:
        # Check if a generation is already in progress
        if not generation_lock.acquire(blocking=False):
            return jsonify({
                'success': False, 
                'error': 'Dataset generation already in progress. Please wait for the current generation to complete.'
            })
        
        try:
            # Find the configuration file
            config_path = Path(__file__).parent / 'dataset_config.yaml'
            
            if not config_path.exists():
                return jsonify({'success': False, 'error': 'Configuration file not found'})
            
            # Get ensure_connected and ensure_unique from request if provided
            ensure_connected = request.json.get('ensure_connected', True) if request.is_json else True
            ensure_unique = request.json.get('ensure_unique', True) if request.is_json else True
            
            # Load and update the configuration
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Set the ensure_connected and ensure_unique options
                    config['ensure_connected'] = ensure_connected
                    config['ensure_unique'] = ensure_unique
                
                # Save the updated configuration
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            except Exception as e:
                print(f"Error updating configuration: {e}")
                # Continue anyway, the default value will be used
            
            # Create the dataset generator with explicit import
            dataset_gen = DatasetGenerator(config_path)
            
            # Check if we need to handle manual generator specially
            if 'generators' in config and 'manual' in config['generators']:
                # Make a copy of the config to avoid modifying the original
                config_copy = dict(config)
                config_copy['generators'] = dict(config['generators'])
                
                # Remove the manual generator from the config copy that will be used for generation
                manual_config = config_copy['generators'].pop('manual', None)
                
                # Update the dataset generator config to use our modified version
                dataset_gen.config = config_copy
            
            # Generate the dataset (this will skip the manual generator now if it was present)
            maps = dataset_gen.generate_dataset()
            
            # Add manually drawn mazes if available in session
            manual_mazes_added = 0
            
            # Check if manual generator is enabled in config
            manual_enabled = config and 'generators' in config and 'manual' in config['generators']
            
            if manual_enabled and 'manual_mazes' in session:
                # Get manual mazes from session
                manual_mazes = session.get('manual_mazes', {})
                
                # Check if manual_mazes is a valid dictionary with content
                if not isinstance(manual_mazes, dict) or not manual_mazes:
                    print(f"Warning: manual_mazes is not a valid dictionary or is empty: {type(manual_mazes)}")
                    # Return early if there are no manual mazes to add
                    if not dataset_gen.dataset:
                        return jsonify({
                            'success': False, 
                            'error': 'No mazes were generated. Please disable the manual generator or add manual mazes first.'
                        })
                else:
                    # Print debug info
                    print(f"Adding manual mazes. Found {len(manual_mazes)} mazes in session.")
                    
                    for maze_id, maze_info in manual_mazes.items():
                        try:
                            print(f"Processing manual maze {maze_id}...")
                            
                            # Validate maze info
                            if not isinstance(maze_info, dict) or 'grid' not in maze_info:
                                print(f"Invalid maze_info format for maze_id {maze_id}: {maze_info}")
                                continue
                            
                            # Convert the grid back to a numpy array
                            grid_data = maze_info.get('grid')
                            if not grid_data:
                                print(f"Empty grid data for maze_id {maze_id}")
                                continue
                                
                            maze_data = np.array(grid_data, dtype=np.int8)
                            
                            # Validate dimensions
                            if maze_data.ndim != 2:
                                print(f"Invalid maze dimensions for maze_id {maze_id}: {maze_data.shape}")
                                continue
                            
                            # Add to dataset
                            dataset_gen.dataset.append(maze_data)
                            
                            # Add metadata
                            width = maze_info.get('width', maze_data.shape[1])
                            height = maze_info.get('height', maze_data.shape[0])
                            
                            metadata = {
                                'generator_type': 'manual',
                                'width': width,
                                'height': height,
                                'shape': maze_data.shape,
                                'creation_time': datetime.now().isoformat(),
                                'manual_id': maze_id
                            }
                            dataset_gen.metadata.append(metadata)
                            manual_mazes_added += 1
                            print(f"Successfully added manual maze {maze_id} with dimensions {width}x{height}")
                        except Exception as e:
                            print(f"Error adding manual maze {maze_id}: {str(e)}")
                    
                    print(f"Successfully added {manual_mazes_added} manual mazes to dataset")
            elif manual_enabled:
                print("Manual generator is enabled but no manual mazes found in session")
                # If manual is the only generator and no mazes, show error
                if len(config.get('generators', {})) == 1 and not dataset_gen.dataset:
                    return jsonify({
                        'success': False, 
                        'error': 'Manual generator is enabled but no manual mazes have been created. Please draw some mazes first or enable other generators.'
                    })
            else:
                print("Manual generator not enabled in config, skipping manual mazes")
            
            # Track deduplication information
            deduplication_info = {
                'enabled': ensure_unique,
                'total_before_deduplication': len(dataset_gen.dataset),
                'total_after_deduplication': len(dataset_gen.dataset)  # Default if no deduplication
            }
            
            # If we should ensure uniqueness, deduplicate the dataset
            if ensure_unique and len(dataset_gen.dataset) > 1:
                print("Ensuring all mazes are unique...")
                original_count = len(dataset_gen.dataset)
                removed_count = dataset_gen.deduplicate_dataset()
                deduplication_info['total_after_deduplication'] = len(dataset_gen.dataset)
                deduplication_info['duplicates_removed'] = removed_count
                print(f"Removed {removed_count} duplicate mazes.")
            
            # Save the dataset
            base_folder = Path(__file__).parent
            dataset_gen.save_dataset(base_folder)
            
            # Get the path to the saved dataset
            dataset_name = dataset_gen.config.get('name', 'default_dataset')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_folder = base_folder / f"{dataset_name}_{timestamp}"
            
            # Wait a moment to ensure the dataset is fully saved
            import time
            time.sleep(0.5)
            
            # Check if the folder exists
            if not dataset_folder.exists():
                # Look for similar folders that might have been created
                potential_folders = list(base_folder.glob(f"{dataset_name}_*"))
                if potential_folders:
                    # Use the most recently created folder
                    dataset_folder = max(potential_folders, key=lambda p: p.stat().st_mtime)
                    
                    if not dataset_folder.exists():
                        return jsonify({
                            'success': False,
                            'error': f'Dataset folder not found: {str(dataset_folder)}'
                        })
            
            # Return success with dataset info
            return jsonify({
                'success': True,
                'dataset_path': str(dataset_folder),
                'dataset_name': dataset_name,
                'map_count': len(dataset_gen.dataset),
                'manual_mazes_added': manual_mazes_added,
                'load_url': url_for('load_dataset'),
                'deduplication_info': deduplication_info
            })
            
        finally:
            # Always release the lock when done
            generation_lock.release()
            
    except Exception as e:
        import traceback
        print(f"Error in generate_full_dataset: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_manual_maze', methods=['POST'])
def save_manual_maze():
    """Save a manually drawn maze to a temporary storage for later use in dataset generation."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        grid_data = data.get('grid_data')
        
        if not grid_data:
            return jsonify({'success': False, 'error': 'No grid data provided'})
        
        # Get the enforce connectivity setting
        enforce_connectivity = data.get('enforce_connectivity', False)
        
        # Convert the grid data to a numpy array
        try:
            maze = np.array(grid_data, dtype=np.int8)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error converting grid data: {str(e)}'})
        
        # If enforce connectivity is enabled, ensure the maze is fully connected
        if enforce_connectivity:
            # Create a temporary dataset to use the ensure_connectivity method
            temp_dataset = Dataset()
            temp_dataset.maps = [maze]
            temp_metadata = {'generator_type': 'manual', 'width': data.get('width'), 'height': data.get('height')}
            temp_dataset.metadata = [temp_metadata]
            
            # Ensure connectivity and use the modified maze
            obstacles_removed = temp_dataset.ensure_connectivity(0)
            maze = temp_dataset.maps[0]
        
        # Generate a unique ID for this maze
        maze_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Store the maze in the session
        if 'manual_mazes' not in session:
            session['manual_mazes'] = {}
        
        # Make sure manual_mazes is a dict, not a list or other type
        if not isinstance(session['manual_mazes'], dict):
            session['manual_mazes'] = {}
            
        # Get grid dimensions
        height, width = maze.shape
        
        # Convert numpy array to a list for JSON serialization
        session['manual_mazes'][maze_id] = {
            'grid': maze.tolist(),
            'width': width,
            'height': height,
            'creation_time': datetime.now().isoformat()
        }
        
        # Make sure session changes are saved
        session.modified = True
        
        # Convert the maze to a base64 image for preview
        maze_img = maze_to_base64(maze)
        
        return jsonify({
            'success': True, 
            'maze_id': maze_id,
            'maze_image': maze_img,
            'dimensions': f"{width}×{height}"
        })
        
    except Exception as e:
        print(f"Error in save_manual_maze: {str(e)}")
        return jsonify({'success': True, 'error': str(e), 'maze_id': 'error', 'maze_image': '', 'dimensions': '0×0'})

@app.route('/clear_manual_mazes', methods=['POST'])
def clear_manual_mazes():
    """Clear all manually drawn mazes from the session."""
    try:
        if 'manual_mazes' in session:
            session.pop('manual_mazes')
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_manual_maze', methods=['POST'])
def delete_manual_maze():
    """Delete a specific manually drawn maze from the saved mazes."""
    try:
        data = request.get_json()
        maze_id = data.get('maze_id')
        
        if not maze_id:
            return jsonify({'success': False, 'error': 'No maze ID provided'})
        
        # Load saved mazes from session
        manual_mazes = session.get('manual_mazes', {})
        
        # Check if manual_mazes is a dictionary
        if not isinstance(manual_mazes, dict):
            return jsonify({'success': False, 'error': 'Invalid manual mazes data structure'})
        
        # Check if the maze ID exists
        if maze_id not in manual_mazes:
            return jsonify({'success': False, 'error': 'Maze not found'})
            
        # Delete the maze from the dictionary
        del manual_mazes[maze_id]
        
        # Update session
        session['manual_mazes'] = manual_mazes
        session.modified = True
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_manual_mazes', methods=['GET'])
def get_manual_mazes():
    """Get all manually drawn mazes from the session."""
    try:
        # Get manual mazes from session
        manual_mazes = session.get('manual_mazes', {})
        
        if not isinstance(manual_mazes, dict):
            return jsonify({'success': False, 'error': 'Invalid manual mazes data'})
        
        # Convert mazes to a format suitable for the frontend
        result_mazes = []
        for maze_id, maze_info in manual_mazes.items():
            try:
                # Convert the grid data to get a preview image
                grid_data = maze_info.get('grid')
                if not grid_data:
                    continue
                    
                maze_data = np.array(grid_data, dtype=np.int8)
                    
                # Get maze dimensions
                width = maze_info.get('width', maze_data.shape[1])
                height = maze_info.get('height', maze_data.shape[0])
                
                # Generate a preview image
                maze_img = maze_to_base64(maze_data)
                
                # Add to result list
                result_mazes.append({
                    'maze_id': maze_id,
                    'maze_image': maze_img,
                    'dimensions': f"{width}×{height}"
                })
            except Exception as e:
                print(f"Error processing maze {maze_id}: {str(e)}")
        
        return jsonify({
            'success': True, 
            'mazes': result_mazes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    template_dir = Path('templates/dataset_builder')
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the index.html template
    index_template = template_dir / 'index.html'
    if not index_template.exists():
        index_template.write_text('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maze Dataset Builder</title>
    <style>
        :root {
            --bg-color: #1e1e1e;
            --text-color: #e0e0e0;
            --accent-color: #64b5f6;
            --panel-bg: #2d2d2d;
            --border-color: #444;
            --button-color: #394e63;
            --button-hover: #4a6583;
            --input-bg: #252525;
            --input-border: #444;
            --card-bg: #2d2d2d;
            --success-color: #4caf50;
            --error-color: #f44336;
        }
        
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: var(--panel-bg);
            color: var(--accent-color);
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 5px;
            border: 1px solid var(--border-color);
        }
        .header h1 {
            margin: 0;
            font-size: 1.8rem;
        }
        .actions-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
        }
        .action-card {
            background-color: var(--card-bg);
            border-radius: 5px;
            padding: 25px;
            width: 300px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .action-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        .action-icon {
            font-size: 36px;
            margin-bottom: 15px;
            color: var(--accent-color);
        }
        .action-title {
            font-size: 20px;
            margin-bottom: 15px;
            color: var(--text-color);
        }
        .action-description {
            color: #999;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .btn {
            display: inline-block;
            background-color: var(--button-color);
            color: var(--text-color);
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            border: 1px solid var(--border-color);
            transition: background-color 0.2s;
            cursor: pointer;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        .btn:hover {
            background-color: var(--button-hover);
        }
        input {
            padding: 10px;
            border: 1px solid var(--input-border);
            border-radius: 4px;
            width: 100%;
            margin-bottom: 10px;
            background-color: var(--input-bg);
            color: var(--text-color);
            font-family: 'Consolas', 'Monaco', monospace;
        }
        .terminal-text {
            color: var(--accent-color);
            margin-bottom: 10px;
            font-size: 14px;
        }
        .terminal-text::before {
            content: "$ ";
            color: var(--success-color);
        }
        
        /* Notification styling */
        .notification {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            animation: slide-in 0.3s ease-out;
        }
        
        @keyframes slide-in {
            0% { transform: translateX(100%); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        .notification-content {
            width: 300px;
        }
        
        .notification-title {
            font-size: 16px;
            font-weight: bold;
            color: var(--accent-color);
            margin-bottom: 8px;
        }
        
        .notification-text {
            font-size: 14px;
            margin-bottom: 12px;
            word-break: break-all;
        }
        
        .notification-actions {
            display: flex;
            justify-content: space-between;
            gap: 10px;
        }
        
        .btn.secondary {
            background-color: transparent;
            border: 1px solid var(--button-color);
        }
        
        .btn.secondary:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>// Maze Dataset Builder</h1>
            <p>Generate, visualize, and manage maze datasets for AI training</p>
        </div>
        
        <div class="terminal-text">Select an option to continue...</div>
        
        <div class="actions-container">
            <div class="action-card">
                <div class="action-icon">🏗️</div>
                <div class="action-title">Create New Dataset</div>
                <div class="action-description">
                    Generate custom mazes with various algorithms and parameters.
                </div>
                <a href="/create_dataset" class="btn">./create_dataset</a>
            </div>
            
            <div class="action-card">
                <div class="action-icon">📂</div>
                <div class="action-title">Load Existing Dataset</div>
                <div class="action-description">
                    Load and visualize an existing dataset from a folder path.
                </div>
                <input type="text" placeholder="/path/to/dataset" id="dataset-path">
                <a href="#" class="btn" id="load-btn">./load_dataset</a>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('load-btn').addEventListener('click', function() {
            const path = document.getElementById('dataset-path').value;
            if (path) {
                this.classList.add('disabled');
                this.textContent = "Loading...";
                setTimeout(() => {
                    alert('Load dataset functionality not implemented yet.\nPath: ' + path);
                    this.classList.remove('disabled');
                    this.textContent = "./load_dataset";
                }, 500);
            } else {
                const inputEl = document.getElementById('dataset-path');
                inputEl.style.borderColor = 'var(--error-color)';
                setTimeout(() => {
                    inputEl.style.borderColor = 'var(--input-border)';
                }, 1500);
            }
        });
        
        // Check for a recently generated dataset path in localStorage
        document.addEventListener('DOMContentLoaded', function() {
            const lastDatasetPath = localStorage.getItem('lastDatasetPath');
            if (lastDatasetPath) {
                // Show a notification
                const notificationEl = document.createElement('div');
                notificationEl.className = 'notification';
                notificationEl.innerHTML = `
                    <div class="notification-content">
                        <div class="notification-title">Dataset Generated!</div>
                        <div class="notification-text">Your dataset was created at: ${lastDatasetPath}</div>
                        <div class="notification-actions">
                            <button class="btn" id="load-generated-dataset">Load Dataset</button>
                            <button class="btn secondary" id="dismiss-notification">Dismiss</button>
                        </div>
                    </div>
                `;
                document.body.appendChild(notificationEl);
                
                // Add button handlers
                document.getElementById('load-generated-dataset').addEventListener('click', function() {
                    // Set the path in the input and submit
                    document.getElementById('dataset-path').value = lastDatasetPath;
                    // Remove from localStorage
                    localStorage.removeItem('lastDatasetPath');
                    // Remove notification
                    notificationEl.remove();
                    
                    // Submit the form to load the dataset
                    const form = document.createElement('form');
                    form.method = 'POST';
                    form.action = '/load_dataset';
                    const input = document.createElement('input');
                    input.type = 'hidden';
                    input.name = 'dataset_path';
                    input.value = lastDatasetPath;
                    form.appendChild(input);
                    document.body.appendChild(form);
                    form.submit();
                });
                
                document.getElementById('dismiss-notification').addEventListener('click', function() {
                    localStorage.removeItem('lastDatasetPath');
                    notificationEl.remove();
                });
            }
        });
    </script>
</body>
</html>
        ''')
    
    # Create the create.html template
    create_template = template_dir / 'create.html'
    if not create_template.exists():
        create_template.write_text('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Dataset - Maze Dataset Builder</title>
    <style>
        :root {
            --bg-color: #1e1e1e;
            --text-color: #e0e0e0;
            --accent-color: #64b5f6;
            --panel-bg: #2d2d2d;
            --border-color: #444;
            --button-color: #394e63;
            --button-hover: #4a6583;
            --input-bg: #252525;
            --input-border: #444;
            --card-bg: #2d2d2d;
            --success-color: #4caf50;
            --error-color: #f44336;
            --disabled-color: #555;
        }
        
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .container {
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 220px;
            background-color: var(--panel-bg);
            color: var(--text-color);
            padding: 15px;
            overflow-y: auto;
            border-right: 1px solid var(--border-color);
        }
        .logo {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 20px;
            color: var(--accent-color);
            text-align: center;
        }
        .main-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }
        .dataset-name-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .dataset-name {
            font-size: 18px;
            padding: 8px 12px;
            border: 1px dashed var(--border-color);
            border-radius: 4px;
            display: inline-block;
            min-width: 200px;
            text-align: center;
            cursor: pointer;
            background-color: var(--input-bg);
            color: var(--accent-color);
        }
        .dataset-name:hover {
            background-color: #333;
        }
        .generator-button {
            display: block;
            padding: 10px 12px;
            margin-bottom: 8px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            text-decoration: none;
            color: var(--text-color);
            border: 1px solid transparent;
            transition: all 0.2s;
        }
        .generator-button:hover, .generator-button.active {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: var(--border-color);
        }
        .generator-button.active {
            color: var(--accent-color);
            border-left: 3px solid var(--accent-color);
        }
        .generator-options {
            background-color: var(--panel-bg);
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border: 1px solid var(--border-color);
        }
        .form-group {
            margin-bottom: 12px;
        }
        .form-label {
            display: block;
            margin-bottom: 4px;
            color: #aaa;
            font-size: 13px;
        }
        .form-select, .form-input {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid var(--input-border);
            border-radius: 4px;
            background-color: var(--input-bg);
            color: var(--text-color);
            font-family: 'Consolas', 'Monaco', monospace;
        }
        .form-select {
            cursor: pointer;
        }
        .form-select:focus, .form-input:focus {
            border-color: var(--accent-color);
            outline: none;
        }
        .options-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
        }
        .generate-btn {
            background-color: var(--button-color);
            color: var(--text-color);
            padding: 10px 15px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        .generate-btn:hover:not(:disabled) {
            background-color: var(--button-hover);
        }
        .generate-btn:disabled {
            background-color: var(--disabled-color);
            cursor: not-allowed;
            opacity: 0.7;
        }
        .maps-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .map-card {
            background-color: var(--panel-bg);
            border-radius: 4px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            text-align: center;
            border: 1px solid var(--border-color);
        }
        .map-image {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            background-color: #222;
        }
        .map-info {
            margin-top: 8px;
            font-size: 12px;
            color: #aaa;
        }
        .section-title {
            font-size: 18px;
            margin-bottom: 12px;
            color: var(--accent-color);
            display: flex;
            align-items: center;
        }
        .section-title::before {
            content: ">";
            margin-right: 8px;
            color: var(--success-color);
        }
        .checkbox-container {
            display: flex;
            align-items: center;
            margin-top: 10px;
        }
        .checkbox-container input {
            margin-right: 8px;
        }
        .back-btn {
            display: inline-block;
            margin-bottom: 15px;
            padding: 6px 12px;
            background-color: transparent;
            color: var(--accent-color);
            text-decoration: none;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            transition: background-color 0.2s;
            font-size: 13px;
        }
        .back-btn:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
        .loading {
            text-align: center;
            padding: 15px;
            font-style: italic;
            color: #999;
        }
        .loading::before {
            content: "> ";
            color: var(--success-color);
        }
        h3 {
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 12px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="logo">Maze Dataset Builder</div>
            <h3>Generators</h3>
            {% for key, gen in generators.items() %}
            <a href="/create_dataset?generator={{ key }}" class="generator-button {% if current_generator == key %}active{% endif %}">
                {{ gen.name }}
            </a>
            {% endfor %}
            <div style="margin-top: 20px;">
                <a href="/" class="generator-button">← Return Home</a>
            </div>
        </div>
        
        <div class="main-content">
            <div class="dataset-name-container">
                <div class="dataset-name" id="dataset-name">{{ dataset_name }}</div>
            </div>
            
            <div class="generator-options">
                <div class="section-title">{{ generator_info.name }} Generator</div>
                <form id="generator-form">
                    <input type="hidden" name="generator_type" value="{{ current_generator }}">
                    
                    <div class="options-grid">
                        <div class="form-group">
                            <label class="form-label">Maze Size</label>
                            <select name="size" class="form-select" id="maze-size">
                                {% for size in maze_sizes %}
                                <option value="{{ size.width }},{{ size.height }}" {% if size.width == 21 %}selected{% endif %}>
                                    {{ size.name }}
                                </option>
                                {% endfor %}
                            </select>
                            <input type="hidden" name="width" id="width-input" value="21">
                            <input type="hidden" name="height" id="height-input" value="21">
                        </div>
                        
                        {% for param_name, param_info in generator_info.params.items() %}
                        <div class="form-group">
                            <label class="form-label">{{ param_info.label }}</label>
                            <input 
                                type="{{ param_info.type }}" 
                                name="{{ param_name }}" 
                                class="form-input" 
                                value="{{ param_info.default }}" 
                                min="{{ param_info.min }}" 
                                max="{{ param_info.max }}" 
                                step="{{ param_info.step }}"
                            >
                        </div>
                        {% endfor %}
                        
                        <div class="form-group">
                            <label class="form-label">Number of Maps</label>
                            <select name="num_samples" class="form-select">
                                <option value="4">4 maps</option>
                                <option value="8" selected>8 maps</option>
                                <option value="12">12 maps</option>
                                <option value="16">16 maps</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="checkbox-container">
                        <input type="checkbox" name="ensure_connected" id="ensure-connected" checked>
                        <label for="ensure-connected">Ensure all mazes are fully connected</label>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <button type="submit" class="generate-btn" id="generate-btn">./generate_maps</button>
                    </div>
                </form>
            </div>
            
            <div id="loading" class="loading" style="display: none;">
                Processing: Generating mazes, please wait...
            </div>
            
            <div id="maps-container" class="maps-container">
                <!-- Generated maps will be displayed here -->
            </div>
        </div>
    </div>
    
    <script>
        // Handle dataset name editing
        const datasetNameElem = document.getElementById('dataset-name');
        datasetNameElem.addEventListener('click', function() {
            const currentName = this.textContent;
            const newName = prompt('Enter a new dataset name:', currentName);
            
            if (newName && newName.trim() !== '') {
                // Send AJAX request to update name
                fetch('/update_dataset_name', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ name: newName.trim() })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.textContent = data.name;
                    }
                });
            }
        });
        
        // Handle maze size selection
        const mazeSizeSelect = document.getElementById('maze-size');
        const widthInput = document.getElementById('width-input');
        const heightInput = document.getElementById('height-input');
        
        mazeSizeSelect.addEventListener('change', function() {
            const [width, height] = this.value.split(',');
            widthInput.value = width;
            heightInput.value = height;
        });
        
        // Initialize size inputs
        const [width, height] = mazeSizeSelect.value.split(',');
        widthInput.value = width;
        heightInput.value = height;
        
        // Handle form submission
        document.getElementById('generator-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const mapsContainer = document.getElementById('maps-container');
            const loading = document.getElementById('loading');
            const generateBtn = document.getElementById('generate-btn');
            
            // Disable the generate button
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            
            // Show loading indicator
            loading.style.display = 'block';
            mapsContainer.innerHTML = '';
            
            // Add the ensure_connected value
            const ensureConnected = document.getElementById('ensure-connected').checked;
            formData.set('ensure_connected', ensureConnected ? 'true' : 'false');
            
            // Send request to generate maps
            fetch('/generate_maps', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.success) {
                    // Display the generated maps
                    data.maps.forEach(map => {
                        const mapCard = document.createElement('div');
                        mapCard.className = 'map-card';
                        
                        const mapImage = document.createElement('img');
                        mapImage.src = 'data:image/png;base64,' + map.image;
                        mapImage.className = 'map-image';
                        mapImage.alt = 'Maze';
                        
                        const mapInfo = document.createElement('div');
                        mapInfo.className = 'map-info';
                        mapInfo.textContent = map.info;
                        
                        mapCard.appendChild(mapImage);
                        mapCard.appendChild(mapInfo);
                        mapsContainer.appendChild(mapCard);
                    });
                } else {
                    // Show error in loading area instead of alert
                    loading.textContent = 'Error: ' + data.error;
                    loading.style.color = 'var(--error-color)';
                    loading.style.display = 'block';
                    setTimeout(() => {
                        loading.style.color = '#999';
                    }, 3000);
                }
                
                // Re-enable the generate button
                generateBtn.disabled = false;
                generateBtn.textContent = './generate_maps';
            })
            .catch(error => {
                loading.style.display = 'block';
                loading.textContent = 'Error: ' + error.message;
                loading.style.color = 'var(--error-color)';
                console.error('Error:', error);
                
                // Re-enable the generate button
                generateBtn.disabled = false;
                generateBtn.textContent = './generate_maps';
                
                setTimeout(() => {
                    loading.style.color = '#999';
                }, 3000);
            });
        });
    </script>
</body>
</html>
        ''')
    
    print("Starting dataset builder web app...")
    print("Access the application at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000) 