import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyBehaviorWeights:
    def __init__(self):
        self._build_fuzzy_system()
        self.rules = self._generate_rules()  # fixed rule set
        self.system = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def _build_fuzzy_system(self):
        # Inputs
        self.distance = ctrl.Antecedent(np.linspace(0, 200, 201), 'distance')
        self.angle = ctrl.Antecedent(np.linspace(-180, 180, 361), 'angle')

        # Outputs
        self.pursuit = ctrl.Consequent(np.linspace(-1, 1, 201), 'pursuit_weight')
        self.fov = ctrl.Consequent(np.linspace(-1, 1, 201), 'fov_weight')
        self.turn = ctrl.Consequent(np.linspace(0, 1, 101), 'turn_weight')

        # Distance MFs
        self.distance['TooClose'] = fuzz.trapmf(self.distance.universe, [0, 0, 40, 60])
        self.distance['Optimal'] = fuzz.trimf(self.distance.universe, [45, 75, 105])
        self.distance['TooFar'] = fuzz.trapmf(self.distance.universe, [85, 110, 200, 200])

        # Angle MFs
        self.angle['ShL'] = fuzz.trapmf(self.angle.universe, [-180, -180, -90, -40])
        self.angle['SoL'] = fuzz.trimf(self.angle.universe, [-90, -40, -10])
        self.angle['AL']  = fuzz.trimf(self.angle.universe, [-20, 0, 20])
        self.angle['SoR'] = fuzz.trimf(self.angle.universe, [10, 40, 90])
        self.angle['ShR'] = fuzz.trapmf(self.angle.universe, [40, 90, 180, 180])

        for output in [self.pursuit, self.fov]:
            output['Negative'] = fuzz.trimf(output.universe, [-1.0, -1.0, 0.0])
            output['Zero'] = fuzz.trimf(output.universe, [-0.2, 0.0, 0.2])
            output['Positive'] = fuzz.trimf(output.universe, [0.0, 1.0, 1.0])

        self.turn['Low'] = fuzz.trimf(self.turn.universe, [0.0, 0.0, 0.5])
        self.turn['Medium'] = fuzz.trimf(self.turn.universe, [0.2, 0.5, 0.8])
        self.turn['High'] = fuzz.trimf(self.turn.universe, [0.5, 1.0, 1.0])

    def _generate_rules(self):
        d, a, t, p, tr = self.distance, self.angle, self.fov, self.pursuit, self.turn
        rules = []
        rules += [
            ctrl.Rule(d['TooClose'] & a['AL'], (t['Zero'], p['Negative'], tr['Low'])),
            ctrl.Rule(d['TooClose'] & a['ShL'], (t['Negative'], p['Negative'], tr['High'])),
            ctrl.Rule(d['TooClose'] & a['ShR'], (t['Negative'], p['Negative'], tr['High'])),

            ctrl.Rule(d['Optimal'] & a['AL'], (t['Positive'], p['Zero'], tr['Low'])),
            ctrl.Rule(d['Optimal'] & a['SoL'], (t['Positive'], p['Zero'], tr['Medium'])),
            ctrl.Rule(d['Optimal'] & a['SoR'], (t['Positive'], p['Zero'], tr['Medium'])),

            ctrl.Rule(d['TooFar'] & a['AL'], (t['Zero'], p['Positive'], tr['Low'])),
            ctrl.Rule(d['TooFar'] & a['SoL'], (t['Positive'], p['Positive'], tr['Medium'])),
            ctrl.Rule(d['TooFar'] & a['SoR'], (t['Positive'], p['Positive'], tr['Medium'])),
            ctrl.Rule(d['TooFar'] & a['ShL'], (t['Zero'], p['Positive'], tr['High'])),
            ctrl.Rule(d['TooFar'] & a['ShR'], (t['Zero'], p['Positive'], tr['High']))
        ]
        return rules

    def evaluate(self, distance_px, angle_rad, is_moving):
        try:
            self.sim.input['distance'] = np.clip(distance_px, 0, 200)
            self.sim.input['angle'] = np.degrees(np.clip(angle_rad, -np.pi, np.pi))
            print(self.sim.input)
            self.sim.compute()
        except Exception as e:
            print("Fuzzy input error:", e)
            raise

        return {
            'fov': self.sim.output.get('fov_weight', 0),
            'pursuit': self.sim.output.get('pursuit_weight', 0),
            'turn': self.sim.output.get('turn_weight', 0)
        }
