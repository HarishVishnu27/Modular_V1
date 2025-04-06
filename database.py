import sqlite3
from datetime import datetime
from config import DATABASE

def init_db(database_name=DATABASE):
    """Initialize database with required tables"""
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    for cam_id in range(1, 5):
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS traffic_data_cam{cam_id} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                r1_vehicle_count INTEGER NOT NULL,
                r1_density REAL NOT NULL,
                weighted_vehicle_count REAL NOT NULL,
                weighted_density REAL NOT NULL,
                vdc{cam_id} BOOLEAN NOT NULL,
                processing_time REAL NOT NULL
            )
        ''')

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS zebra_crossing_data_cam{cam_id} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                x_position INTEGER NOT NULL,
                y_position INTEGER NOT NULL,
                road_side TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        ''')

    conn.commit()
    conn.close()

def save_traffic_data(cam_id, r1_vehicles, r1_density, weighted_vehicles, weighted_density, vdc, processing_time):
    """Save traffic data to database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(f'''INSERT INTO traffic_data_cam{cam_id}
                       (timestamp, r1_vehicle_count, r1_density, 
                        weighted_vehicle_count, weighted_density, 
                        vdc{cam_id}, processing_time)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (timestamp, r1_vehicles, r1_density,
                     weighted_vehicles, weighted_density,
                     vdc, processing_time))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Database error for camera {cam_id}: {e}")
        return False

def save_zebra_crossing_data(cam_id, vehicle_type, x_position, y_position, road_side, confidence):
    """Save zebra crossing detection data"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(f'''INSERT INTO zebra_crossing_data_cam{cam_id}
                           (timestamp, vehicle_type, x_position, y_position, road_side, confidence)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                       (timestamp, vehicle_type, x_position, y_position, road_side, confidence))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Database error for camera {cam_id}: {e}")
        return False

def get_traffic_data(cam_id, start_date, end_date, limit=100):
    """Get traffic data for a camera within a date range"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute(f'''SELECT 
                id, timestamp, r1_vehicle_count, r1_density,
                weighted_vehicle_count, weighted_density, 
                vdc{cam_id}, processing_time
                FROM traffic_data_cam{cam_id} 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC LIMIT ?''',
                (start_date, end_date, limit))
        
        rows = cursor.fetchall()
        conn.close()
        return rows if rows else []
    except sqlite3.Error as e:
        print(f"Error fetching data for camera {cam_id}: {e}")
        return []

def get_zebra_crossing_data(cam_id, limit=500):
    """Get zebra crossing data for a camera"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        query = f'''
            SELECT 
                timestamp, vehicle_type, 
                COUNT(*) as vehicle_count
            FROM zebra_crossing_data_cam{cam_id}
            GROUP BY timestamp, vehicle_type
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        cursor.execute(query)
        rows = cursor.fetchall()
        
        processed_rows = [
            {
                'timestamp': row[0],
                'vehicle_type': row[1],
                'vehicle_count': row[2]
            } for row in rows
        ]
        
        conn.close()
        return processed_rows
    except sqlite3.Error as e:
        print(f"Error fetching zebra crossing data for camera {cam_id}: {e}")
        return []

def get_camera_stats(camera_id):
    """Get latest statistics for a camera"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute(f'''SELECT 
                        timestamp,
                        r1_vehicle_count,
                        r1_density,
                        weighted_vehicle_count,
                        weighted_density
                    FROM traffic_data_cam{camera_id} 
                    ORDER BY timestamp DESC LIMIT 1''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'timestamp': result[0],
                'vehicle_count': result[1],
                'density': float(result[2]),
                'total': {
                    'vehicle_count': result[3],
                    'density': float(result[4])
                }
            }
        else:
            return {'status': 'No data available'}
            
    except sqlite3.Error as e:
        print(f"Database error for camera {camera_id}: {e}")
        return {'error': str(e), 'status': 'Database error'}

def get_vehicle_detect_status(camera_id):
    """Get vehicle detection status for a camera"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute(f'''SELECT vdc{camera_id}
                    FROM traffic_data_cam{camera_id} 
                    ORDER BY timestamp DESC LIMIT 1''')
        
        result = cursor.fetchone()
        conn.close()
        
        return 1 if result and result[0] else 0
        
    except sqlite3.Error as e:
        print(f"Database error for camera {camera_id}: {e}")
        return 0