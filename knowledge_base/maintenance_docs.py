"""
Synthetic maintenance manual text chunks for RAG.
"""

maintenance_chunks = [
    # Bearing Faults
    "Bearing fault detected due to high vibration and abnormal acoustic noise. Step 1: Immediately shut down the machine to prevent catastrophic failure.",
    "For bearing faults, inspect the lubrication system. Ensure that the correct grade of industrial grease is applied. Relubricate every 500 hours of operation.",
    "If vibration exceeds 15 Hz for a sustained period, the main rotor bearings must be replaced. Refer to section 4.2 of the replacement manual.",
    "Bearing replacement procedure: Isolate power, remove outer casing, extract worn bearing using a hydraulic puller, press fit the new bearing.",
    "After installing a new bearing, run the machine at 20% RPM for 30 minutes to ensure proper seating before ramping up to full speed.",
    "A bearing fault is characterized by a sudden spike in RMS vibration. Check for metal shavings in the oil pan which confirm bearing degradation.",
    
    # Overheating
    "Overheating is often caused by a blocked cooling intake. Stop the machine, allow it to cool for 2 hours, and clean all cooling fins.",
    "If temperature exceeds 90°C, the thermal trip circuit will engage. Verify coolant fluid levels and check for leaks in the primary reservoir.",
    "Overheating cooldown procedure: Do not use water to rapidly cool components. Use industrial fans to gradually reduce temperature.",
    "Check the thermal sensor calibration if overheating alarms are frequent but physical inspection shows no abnormal heat.",
    "Persistent overheating may indicate excessive friction in the drive shaft. Apply high-temp synthetic oil to the main friction points.",
    "Ensure ambient temperature in the facility does not exceed 40°C, as this reduces the effectiveness of the machine's cooling system.",
    
    # Pressure Anomalies
    "Pressure drops usually indicate a leak in the pneumatic seals. Inspect all O-rings and replace any that are cracked or brittle.",
    "If pressure spikes above 2.0 bar, the pressure relief valve may be stuck. Manually actuate the valve to clear debris.",
    "Pressure anomaly troubleshooting: Isolate the compressor line, attach a manual gauge, and compare readings to identify sensor drift.",
    "For sustained low pressure, check the main compressor filter. A clogged filter will starve the system of air intake.",
    "Recalibrate the pressure transducer every 6 months to ensure accurate readings. Use a certified reference standard.",
    "High pressure coupled with low RPM indicates a system blockage. Flush the lines with industrial solvent before restarting.",
    
    # Preventive Maintenance
    "Preventive maintenance schedule: Weekly check of fluid levels, monthly inspection of drive belts, quarterly calibration of all sensors.",
    "To maximize machine lifespan, perform a full deep clean of the chassis and internal components every 6 months.",
    "Replace all primary filters (air, oil, coolant) after 2000 hours of operation regardless of their visual condition.",
    "Log all minor anomalies in the maintenance ledger. A pattern of minor anomalies often precedes a major fault.",
    "During preventive maintenance, torque all accessible bolts to the manufacturer specified tightness to prevent vibration-induced loosening.",
    "Update the machine's firmware annually to ensure compatibility with the latest central monitoring software.",
    
    # Safety Procedures
    "Safety shutdown procedure: Press the emergency stop button, turn off the main breaker, and apply a lockout/tagout (LOTO) device.",
    "Never bypass safety interlocks. Bypassing interlocks during operation can result in severe injury and voids the equipment warranty.",
    "When diagnosing active electrical faults, technicians must wear Class 2 insulated gloves and use insulated tools.",
    "In the event of an uncontrolled fire, do not use water on electrical or oil-based fires. Use a Class C fire extinguisher.",
    "Before restarting after an emergency shutdown, a senior supervisor must visually inspect and sign off on the machine's condition.",
    "If toxic fumes are detected (e.g., burning electrical insulation), evacuate the immediate area and activate the facility's exhaust system."
]
