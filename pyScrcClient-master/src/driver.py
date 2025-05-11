import msgParser
import carState
import carControl
import numpy as np
import joblib
import os
import logging

class Driver(object):
    '''
    A driver object for the SCRC that uses a trained ML model
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        
        # Add gear shifting protection
        self.gear_change_timeout = 20  # Frames to wait before allowing another gear change
        self.gear_change_counter = 0
        self.prev_gear = 0
        
        # Setup logging
        logging.basicConfig(filename='driver_ml.log', level=logging.INFO,
                           format='%(asctime)s - %(levelname)s: %(message)s')
        
        # Load ML model and scaler
        try:
            self.model_path = os.path.join(os.path.dirname(__file__), 'torcs_mlp_model.joblib')
            self.scaler_path = os.path.join(os.path.dirname(__file__), 'torcs_scaler.joblib')
            
            # Try to load both the combined model and individual models if they exist
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logging.info("Successfully loaded combined ML model.")
                self.using_separate_models = False
            else:
                # Try loading individual models for each control
                model_files = {
                    'accel': 'torcs_accel_model.joblib',
                    'brake': 'torcs_brake_model.joblib',
                    'clutch': 'torcs_clutch_model.joblib',
                    'steer': 'torcs_steer_model.joblib'
                }
                self.models = {}
                
                for control, filename in model_files.items():
                    model_path = os.path.join(os.path.dirname(__file__), filename)
                    if os.path.exists(model_path):
                        self.models[control] = joblib.load(model_path)
                
                if self.models:
                    logging.info(f"Successfully loaded separate ML models: {list(self.models.keys())}")
                    self.using_separate_models = True
                else:
                    logging.warning("No ML models were found. Using fallback driving strategy.")
                    self.using_separate_models = False
                    self.model = None
                    
                self.scaler = joblib.load(self.scaler_path) if os.path.exists(self.scaler_path) else None
            
        except Exception as e:
            logging.error(f"Error loading ML model: {str(e)}")
            self.model = None
            self.using_separate_models = False
            self.scaler = None
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        # Use ML model if available, otherwise fall back to rule-based driving
        if self.model is not None or self.using_separate_models:
            try:
                # Prepare features for model input
                features = self.extract_features()
                
                if self.using_separate_models:
                    # Use separate models for each control
                    if 'steer' in self.models:
                        scaled_features = self.scaler.transform([features]) if self.scaler else np.array([features])
                        steer_prediction = self.models['steer'].predict(scaled_features)[0]
                        self.control.setSteer(steer_prediction)
                    else:
                        self.steer()  # Fallback
                        
                    if 'accel' in self.models:
                        scaled_features = self.scaler.transform([features]) if self.scaler else np.array([features])
                        accel_prediction = self.models['accel'].predict(scaled_features)[0]
                        self.control.setAccel(accel_prediction)
                    else:
                        self.speed()  # Fallback
                    
                    if 'brake' in self.models:
                        scaled_features = self.scaler.transform([features]) if self.scaler else np.array([features])
                        brake_prediction = self.models['brake'].predict(scaled_features)[0]
                        self.control.setBrake(brake_prediction)
                    
                    if 'clutch' in self.models:
                        scaled_features = self.scaler.transform([features]) if self.scaler else np.array([features])
                        clutch_prediction = self.models['clutch'].predict(scaled_features)[0]
                        self.control.setClutch(clutch_prediction)
                    
                    # For gear, still use rule-based approach
                    self.improved_gear()
                else:
                    # Use combined model for all controls
                    scaled_features = self.scaler.transform([features]) if self.scaler else np.array([features])
                    predictions = self.model.predict(scaled_features)[0]
                    
                    # Apply predictions to control
                    self.control.setAccel(predictions[0])
                    self.control.setBrake(predictions[1])
                    self.control.setClutch(predictions[2])
                    self.control.setSteer(predictions[3])
                    
                    # For gear, still use rule-based approach
                    self.improved_gear()
                
                logging.debug(f"ML Prediction: accel={self.control.getAccel()}, brake={self.control.getBrake()}, "
                             f"clutch={self.control.getClutch()}, steer={self.control.getSteer()}")
            
            except Exception as e:
                logging.error(f"Error using ML model: {str(e)}. Falling back to rule-based driving.")
                # Fall back to rule-based driving if model fails
                self.steer()
                self.improved_gear()
                self.speed()
        else:
            # Use rule-based driving if no model is available
            self.steer()
            self.improved_gear()
            self.speed()
        
        return self.control.toMsg()
    
    def extract_features(self):
        """Extract features from car state in the same format as the training data"""
        features = []
        
        # Add all available sensor data as features
        # This should match the format expected by the model
        features.append(self.state.curLapTime)
        features.append(self.state.angle)
        features.append(self.state.damage)
        features.append(self.state.distFromStart)
        features.append(self.state.distRaced)
        features.append(self.state.fuel)
        features.append(self.state.gear)
        features.append(self.state.lastLapTime)
        features.append(self.state.racePos)
        features.append(self.state.rpm)
        features.append(self.state.speedX)
        features.append(self.state.speedY)
        features.append(self.state.speedZ)
        features.append(self.state.trackPos)
        features.append(self.state.z)
        
        # Add focus sensors
        for i in range(5):  # Assuming 5 focus sensors as in the data example
            if i < len(self.state.focus):
                features.append(self.state.focus[i])
            else:
                features.append(-1.0)  # Default value if sensor not available
        
        # Add opponent sensors
        for i in range(36):  # Assuming 36 opponent sensors as in the data example
            if i < len(self.state.opponents):
                features.append(self.state.opponents[i])
            else:
                features.append(200.0)  # Default value if sensor not available
        
        # Add track sensors
        for i in range(19):  # Assuming 19 track sensors as in the data example
            if i < len(self.state.track):
                features.append(self.state.track[i])
            else:
                features.append(0.0)  # Default value if sensor not available
        
        # Add wheelspinvel sensors
        for i in range(4):  # Assuming 4 wheelspinvel sensors as in the data example
            if i < len(self.state.wheelSpinVel):
                features.append(self.state.wheelSpinVel[i])
            else:
                features.append(0.0)  # Default value if sensor not available
        
        # Calculate additional engineered features similar to what we did in model.py
        if hasattr(self.state, 'speedX') and hasattr(self.state, 'speedY') and hasattr(self.state, 'speedZ'):
            speed_magnitude = np.sqrt(self.state.speedX**2 + self.state.speedY**2 + self.state.speedZ**2)
            features.append(speed_magnitude)
        
        # Calculate average track sensor value
        if hasattr(self.state, 'track') and len(self.state.track) > 0:
            avg_track = sum(self.state.track) / len(self.state.track)
            features.append(avg_track)
        
        # Calculate average opponent distance
        if hasattr(self.state, 'opponents') and len(self.state.opponents) > 0:
            avg_opponent = sum(self.state.opponents) / len(self.state.opponents)
            features.append(avg_opponent)
        
        return features
    
    def steer(self):
        """Traditional steering logic as fallback"""
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def improved_gear(self):
        """Improved gear shifting logic to prevent oscillation between neutral and 1st gear"""
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed_x = self.state.getSpeedX()
        
        # Decrement gear change counter if it's active
        if self.gear_change_counter > 0:
            self.gear_change_counter -= 1
        
        # Only change gears if timeout has passed
        if self.gear_change_counter == 0:
            # Handle neutral gear case specially
            if gear == 0:
                # Always shift to first gear from neutral if not moving backward
                if speed_x >= -5.0:
                    self.control.setGear(1)
                    self.gear_change_counter = self.gear_change_timeout
            else:
                # Determine if we need to shift up or down based on RPM
                if rpm > 7000:
                    # Shift up
                    self.control.setGear(gear + 1)
                    self.gear_change_counter = self.gear_change_timeout
                elif rpm < 3000 and gear > 1:  # Don't downshift to neutral
                    # Shift down but not to neutral
                    self.control.setGear(gear - 1)
                    self.gear_change_counter = self.gear_change_timeout
                # Special case for first gear
                elif gear == 1 and rpm < 800:
                    # Apply more gas in first gear if RPM is too low
                    accel = self.control.getAccel()
                    self.control.setAccel(min(accel + 0.3, 1.0))  # Give more gas
                    self.control.setClutch(0.5)  # Use clutch to avoid stalling
        
        # Keep track of previous gear and RPM
        self.prev_gear = gear
        self.prev_rpm = rpm
    
    def speed(self):
        """Traditional speed control logic as fallback with improved acceleration"""
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        gear = self.state.getGear()
        
        # Apply more aggressive acceleration at low speeds, especially in first gear
        if speed < 10 and gear <= 1:
            accel = 1.0  # Full acceleration for starting
        elif speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
            
    def onShutDown(self):
        logging.info("Driver shutting down")
        pass
    
    def onRestart(self):
        logging.info("Driver restarting")
        self.prev_rpm = None
