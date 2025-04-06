from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session
from functools import wraps
import os
import json
from datetime import datetime, timedelta
import csv
from io import StringIO
import numpy as np

from config import (
    PROCESSED_FOLDER, REGIONS, COLOR_MAPPING, REGIONS_FILE,
    DEFAULT_USERNAME, DEFAULT_PASSWORD, FRAME_SKIP, VEHICLE_THRESHOLD
)
import database
import vision_processing

app = Flask(__name__)
# Secret key for session
app.secret_key = 'harish@2704'

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return "Invalid credentials. Try again.", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    latest_images = {}
    for cam_id in range(1, 5):
        cam_folder = os.path.join(PROCESSED_FOLDER, f'cam{cam_id}')
        if os.path.exists(cam_folder):
            files = [f for f in os.listdir(cam_folder) if f.endswith('.jpg')]
            if files:
                latest_images[f'cam{cam_id}'] = files[-1]
            else:
                latest_images[f'cam{cam_id}'] = None
        else:
            latest_images[f'cam{cam_id}'] = None

    return render_template('index.html', latest_images=latest_images)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if request.method == 'POST':
        data = request.json
        cam_id = data.get('cam_id')
        regions = data.get('regions')

        if cam_id and regions:
            REGIONS[f'cam{cam_id}'] = regions
            return jsonify({'status': 'success', 'message': 'Regions updated'}), 200
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    return render_template('admin.html')

@app.route('/update_regions', methods=['POST'])
def update_regions():
    try:
        data = request.json
        cam_id = data.get('cam_id')
        new_regions = data.get('regions')

        if not cam_id or not new_regions:
            return jsonify({'status': 'error', 'message': 'Missing required data'}), 400

        if cam_id not in REGIONS:
            REGIONS[cam_id] = {}

        for region in new_regions:
            region_type = region.get('type')
            vertices = region.get('vertices')
            color = region.get('color', 'green')

            if not region_type or not vertices:
                continue

            if region_type == 'Zebra':
                REGIONS[cam_id]['Zebra'] = {
                    'vertices': np.array(vertices, dtype=np.int32),
                    'color': COLOR_MAPPING.get(color.lower(), (0, 0, 255)),
                    'weight': 1.0
                }
            elif region_type == 'R':
                REGIONS[cam_id]['R1'] = {
                    'vertices': np.array(vertices, dtype=np.int32),
                    'color': COLOR_MAPPING.get(color.lower(), (0, 255, 0)),
                    'weight': 1.0
                }

        with open(REGIONS_FILE, 'w') as file:
            json_regions = {}
            for cam, regions in REGIONS.items():
                json_regions[cam] = {}
                for reg_name, reg_data in regions.items():
                    json_regions[cam][reg_name] = {
                        'vertices': reg_data['vertices'].tolist(),
                        'color': reg_data['color'],
                        'weight': reg_data['weight']
                    }
            json.dump(json_regions, file, indent=4)

        return jsonify({
            'status': 'success',
            'message': f'Regions updated successfully for {cam_id}'
        }), 200

    except Exception as e:
        print(f"Error updating regions: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error updating regions: {str(e)}'
        }), 500

@app.route('/get_regions', methods=['GET'])
def get_regions():
    try:
        return jsonify(REGIONS), 200
    except Exception as e:
        print(f"Error fetching regions: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_frame/<cam_id>', methods=['GET'])
def get_frame(cam_id):
    try:
        frame = vision_processing.get_single_frame(cam_id)
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to capture frame'}), 500

        filename = f'static/temp/{cam_id}_frame.jpg'
        import cv2
        cv2.imwrite(filename, frame)

        return jsonify({'status': 'success', 'frame_url': f'/{filename}'})
    except Exception as e:
        print(f"Error extracting frame for {cam_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/zebra_crossing_analytics')
@login_required
def zebra_crossing_analytics():
    try:
        data = {}
        for cam_id in range(1, 5):
            data[f'cam{cam_id}'] = database.get_zebra_crossing_data(cam_id)

        return render_template('zebra_crossing.html', data=data)

    except Exception as e:
        print(f"Error in zebra crossing analytics route: {e}")
        return f"Error loading zebra crossing analytics: {str(e)}", 500

@app.route('/analytics')
@login_required
def analytics():
    try:
        # Get date parameters
        start_date = request.args.get('start_date',
            (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date',
            datetime.now().strftime('%Y-%m-%d'))

        # Add one day to end_date to include the entire day
        end_date_inclusive = (datetime.strptime(end_date, '%Y-%m-%d') +
            timedelta(days=1)).strftime('%Y-%m-%d')

        data = {}
        for cam_id in range(1, 5):
            data[f'cam{cam_id}'] = database.get_traffic_data(cam_id, start_date, end_date_inclusive)

        # Calculate aggregates
        aggregates = {}
        for cam_id in range(1, 5):
            cam_data = data[f'cam{cam_id}']
            if cam_data:
                weighted_densities = [row[4] for row in cam_data]
                vehicle_counts = [row[3] for row in cam_data]
                aggregates[f'cam{cam_id}'] = {
                    'avg_density': sum(weighted_densities) / len(weighted_densities)
                        if weighted_densities else 0,
                    'peak_count': max(vehicle_counts) if vehicle_counts else 0
                }
            else:
                aggregates[f'cam{cam_id}'] = {
                    'avg_density': 0,
                    'peak_count': 0
                }

        return render_template('analytics.html',
            data=data,
            aggregates=aggregates,
            start_date=start_date,
            end_date=end_date)

    except Exception as e:
        print(f"Error in analytics route: {e}")
        return f"Error loading analytics: {str(e)}", 500

@app.route('/download_analytics')
@login_required
def download_analytics():
    try:
        start_date = request.args.get('start_date',
            (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date',
            datetime.now().strftime('%Y-%m-%d'))
        end_date_inclusive = (datetime.strptime(end_date, '%Y-%m-%d') +
            timedelta(days=1)).strftime('%Y-%m-%d')

        # Create a CSV string buffer
        si = StringIO()
        cw = csv.writer(si)

        # Write headers
        cw.writerow(['Camera', 'ID', 'Timestamp', 'Vehicle Count', 'Density (%)',
            'Weighted Count', 'Weighted Density (%)', 'VDC', 'Processing Time (ms)'])

        # Fetch and write data for each camera
        for cam_id in range(1, 5):
            rows = database.get_traffic_data(cam_id, start_date, end_date_inclusive, limit=None)
            for row in rows:
                cw.writerow([f'Camera {cam_id}'] + list(row))

        # Create the response
        output = si.getvalue()
        si.close()

        return Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                f"attachment;filename=traffic_analytics_{start_date}_to_{end_date}.csv"
            }
        )

    except Exception as e:
        print(f"Error in download route: {e}")
        return f"Error downloading analytics: {str(e)}", 500

@app.route('/CameraStats', methods=['GET'])
def camera_stats():
    try:
        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cameras': {}
        }

        for camera_id in range(1, 5):
            response['cameras'][f'cam{camera_id}'] = database.get_camera_stats(camera_id)

        return jsonify(response)

    except Exception as e:
        print(f"Server error in camera_stats: {e}")
        return jsonify({
            'error': str(e),
            'status': 'Server error',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/VehicleDetect', methods=['GET'])
def vehicle_detect():
    try:
        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        for camera_id in range(1, 5):
            response[f'vdc{camera_id}'] = database.get_vehicle_detect_status(camera_id)

        return jsonify(response)

    except Exception as e:
        print(f"Server error in vehicle_detect: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'vdc1': 0,
            'vdc2': 0,
            'vdc3': 0,
            'vdc4': 0
        }), 500

@app.route('/get_last_processed/<cam_id>')
def get_last_processed(cam_id):
    try:
        processed_folder = os.path.join(PROCESSED_FOLDER, cam_id)
        if not os.path.exists(processed_folder):
            return jsonify({'status': 'error', 'message': 'No processed images found'}), 404

        files = [f for f in os.listdir(processed_folder) if f.endswith('.jpg')]
        if not files:
            return jsonify({'status': 'error', 'message': 'No processed images found'}), 404

        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(processed_folder, x)))
        return jsonify({'status': 'success', 'image_url': f'/static/processed/{cam_id}/{latest_file}'})

    except Exception as e:
        print(f"Error getting last processed image for {cam_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_frame_skip', methods=['POST'])
def update_frame_skip():
    global FRAME_SKIP
    frame_skip = request.json.get('frame_skip')
    try:
        import config
        config.FRAME_SKIP = int(frame_skip)
        return jsonify({'status': 'success', 'message': 'Frame skip count updated successfully'})
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid frame skip value provided'}), 400

@app.route('/update_vehicle_threshold', methods=['POST'])
def update_vehicle_threshold():
    global VEHICLE_THRESHOLD
    threshold = request.json.get('threshold')
    try:
        import config
        config.VEHICLE_THRESHOLD = int(threshold)
        return jsonify({'status': 'success', 'message': 'Vehicle threshold updated successfully'})
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid threshold value provided'}), 400

@app.route('/get_parameters', methods=['GET'])
@login_required
def get_parameters():
    return jsonify({
        'frame_skip': FRAME_SKIP,
        'vehicle_threshold': VEHICLE_THRESHOLD
    })