DEFINE_ROI = False 

ROI_ZONES = [[(284, 257), (478, 194), (577, 245), (320, 334)], [(512, 187), (686, 139), (821, 181), (587, 235)], [(326, 339), (502, 279), (547, 410), (390, 482)], [(787, 335), (1044, 252), (1386, 400), (1119, 550)], [(399, 485), (773, 345), (1104, 565), (617, 837)]]

# --- Prohance Specific Thresholds ---
# Max duration (seconds) for an interaction to be flagged as potential manipulation.
SHORT_INTERACTION_THRESHOLD_SEC = 10

# Min duration (seconds) for a foreign user's presence to be considered "sustained" occupancy.
SUSTAINED_PRESENCE_THRESHOLD_SEC = 60

# Period after flagging manipulation during which new flags for the same ROI are suppressed.
MANIPULATION_COOLDOWN_SEC = 30