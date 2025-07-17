#!/usr/bin/env python3
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define fuzzy variables
distance = ctrl.Antecedent(np.linspace(0, 200, 201), 'distance')
angle = ctrl.Antecedent(np.linspace(-180, 180, 361), 'angle')

pursuit = ctrl.Consequent(np.linspace(-1, 1, 201), 'pursuit_weight')
track = ctrl.Consequent(np.linspace(-1, 1, 201), 'track_weight')
turn = ctrl.Consequent(np.linspace(0, 1, 101), 'turn_weight')

# Membership functions for distance
distance['TooClose'] = fuzz.trapmf(distance.universe, [0, 0, 40, 65])
distance['Optimal'] = fuzz.trimf(distance.universe, [45, 75, 105])
distance['TooFar'] = fuzz.trapmf(distance.universe, [85, 110, 200, 200])

# Membership functions for angle
angle['ShL'] = fuzz.trapmf(angle.universe, [-np.pi, -np.pi, -1.5, -0.7])
angle['SoL'] = fuzz.trimf(angle.universe, [-1.3, -0.7, -0.2])
angle['AL']  = fuzz.trimf(angle.universe, [-0.4, 0.0, 0.4])
angle['SoR'] = fuzz.trimf(angle.universe, [0.2, 0.7, 1.3])
angle['ShR'] = fuzz.trapmf(angle.universe, [0.7, 1.5, np.pi, np.pi])


# Output membership functions
for output in [pursuit, track]:
    output['Negative'] = fuzz.trimf(output.universe, [-1.0, -1.0, 0.0])
    output['Zero']     = fuzz.trimf(output.universe, [-0.2, 0.0, 0.2])
    output['Positive'] = fuzz.trimf(output.universe, [0.0, 1.0, 1.0])

turn['Low'] = fuzz.trimf(turn.universe, [0.0, 0.0, 0.5])
turn['Medium'] = fuzz.trimf(turn.universe, [0.2, 0.5, 0.8])
turn['High'] = fuzz.trimf(turn.universe, [0.5, 1.0, 1.0])


# 5. Rules
def _generate_rules(self, is_moving):
    self.rules.clear()
    if is_moving:
        R = self.rules
        d = self.distance
        a = self.angle
        t = self.track
        p = self.pursuit
        tr = self.turn
        R.append(ctrl.Rule(d['TooClose'] & a['AL'], (t['Zero'], p['Negative'], tr['Low'])))
        R.append(ctrl.Rule(d['TooClose'] & a['ShL'], (t['Negative'], p['Negative'], tr['High'])))
        R.append(ctrl.Rule(d['TooClose'] & a['ShR'], (t['Negative'], p['Negative'], tr['High'])))
        R.append(ctrl.Rule(d['Optimal'] & a['AL'], (t['Positive'], p['Zero'], tr['Low'])))
        R.append(ctrl.Rule(d['Optimal'] & a['SoL'], (t['Positive'], p['Zero'], tr['Medium'])))
        R.append(ctrl.Rule(d['Optimal'] & a['SoR'], (t['Positive'], p['Zero'], tr['Medium'])))
        R.append(ctrl.Rule(d['Optimal'] & a['ShL'], (t['Zero'], p['Zero'], tr['High'])))
        R.append(ctrl.Rule(d['Optimal'] & a['ShR'], (t['Zero'], p['Zero'], tr['High'])))
        R.append(ctrl.Rule(d['TooFar'] & a['AL'], (t['Zero'], p['Positive'], tr['Low'])))
        R.append(ctrl.Rule(d['TooFar'] & a['SoL'], (t['Positive'], p['Positive'], tr['Medium'])))
        R.append(ctrl.Rule(d['TooFar'] & a['SoR'], (t['Positive'], p['Positive'], tr['Medium'])))
        R.append(ctrl.Rule(d['TooFar'] & a['ShL'], (t['Zero'], p['Positive'], tr['High'])))
        R.append(ctrl.Rule(d['TooFar'] & a['ShR'], (t['Zero'], p['Positive'], tr['High'])))
    else:
        R = self.rules
        d = self.distance
        t = self.track
        p = self.pursuit
        tr = self.turn
        R.append(ctrl.Rule(d['TooClose'], (t['Zero'], p['Negative'], tr['Low'])))
        R.append(ctrl.Rule(d['Optimal'], (t['Zero'], p['Zero'], tr['Low'])))
        R.append(ctrl.Rule(d['TooFar'], (t['Zero'], p['Positive'], tr['Low'])))

# Rule generator
def generate_rules(is_moving):
    rules = []
    d = distance
    a = angle
    t = track
    p = pursuit
    tr = turn

    if is_moving:
        rules.append(ctrl.Rule(d['TooClose'] & a['AL'], (t['Zero'], p['Negative'], tr['Low'])))
        rules.append(ctrl.Rule(d['TooClose'] & a['ShL'], (t['Negative'], p['Negative'], tr['High'])))
        rules.append(ctrl.Rule(d['TooClose'] & a['ShR'], (t['Negative'], p['Negative'], tr['High'])))
        rules.append(ctrl.Rule(d['Optimal'] & a['AL'], (t['Positive'], p['Zero'], tr['Low'])))
        rules.append(ctrl.Rule(d['Optimal'] & a['SoL'], (t['Positive'], p['Zero'], tr['Medium'])))
        rules.append(ctrl.Rule(d['Optimal'] & a['SoR'], (t['Positive'], p['Zero'], tr['Medium'])))
        rules.append(ctrl.Rule(d['Optimal'] & a['ShL'], (t['Zero'], p['Zero'], tr['High'])))
        rules.append(ctrl.Rule(d['Optimal'] & a['ShR'], (t['Zero'], p['Zero'], tr['High'])))
        rules.append(ctrl.Rule(d['TooFar'] & a['AL'], (t['Zero'], p['Positive'], tr['Low'])))
        rules.append(ctrl.Rule(d['TooFar'] & a['SoL'], (t['Positive'], p['Positive'], tr['Medium'])))
        rules.append(ctrl.Rule(d['TooFar'] & a['SoR'], (t['Positive'], p['Positive'], tr['Medium'])))
        rules.append(ctrl.Rule(d['TooFar'] & a['ShL'], (t['Zero'], p['Positive'], tr['High'])))
        rules.append(ctrl.Rule(d['TooFar'] & a['ShR'], (t['Zero'], p['Positive'], tr['High'])))
    else:
        rules.append(ctrl.Rule(d['TooClose'], (t['Zero'], p['Negative'], tr['Low'])))
        rules.append(ctrl.Rule(d['Optimal'], (t['Zero'], p['Zero'], tr['Low'])))
        rules.append(ctrl.Rule(d['TooFar'], (t['Zero'], p['Positive'], tr['Low'])))

    return rules

# Build fuzzy system
rules = generate_rules(is_moving=True)
system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# Plot membership functions
distance.view()
plt.title("Distance Membership Functions")
plt.savefig("distance_mf.png")

angle.view()
plt.title("Angle Membership Functions")
plt.savefig("angle_mf.png")

track.view()
plt.title("Track Weight Membership Functions")
plt.savefig("track_weight_mf.png")

pursuit.view()
plt.title("Pursuit Weight Membership Functions")
plt.savefig("pursuit_weight_mf.png")

turn.view()
plt.title("Turn Weight Membership Functions")
plt.savefig("turn_weight_mf.png")

# Test
test_distance = 70
test_angle = -0.9

sim.input['distance'] = test_distance
sim.input['angle'] = test_angle

sim.compute()

print("Results for:")
print(f"Distance = {test_distance} px")
print(f"Angle = {test_angle} rad ({np.degrees(test_angle):.2f} deg)")
print(f"Track Weight  = {sim.output.get('track_weight', 'N/A')}")
print(f"Pursuit Weight = {sim.output.get('pursuit_weight', 'N/A')}")
print(f"Turn Weight    = {sim.output.get('turn_weight', 'N/A')}")
