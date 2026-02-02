split_objects = {
  'train': {
    'exclude_objects': ['mug', 'pan', 'wine_glass'],
  },
  'test': {
    'include_objects': ['scissors']
    #'apple', 'banana', 'binoculars', 'bowl', 'camera', 'cell_phone', 'cup', 'door_knob', 'eyeglasses', 'flashlight', 'hammer', 'headphones', 'knife', 'light_bulb', 'mouse', 'mug', 'pan', 'ps_controller', 'scissors', 'stapler', 'toothbrush', 'toothpaste', 'utah_teapot', 'water_bottle', 'wine_glass'
  }
}
# 'hammer', 'headphones', 'knife', 'light_bulb', 'mouse', 'mug', 'pan', 'ps_controller', 'scissors', 'stapler', 'toothbrush', 'toothpaste'
#, 'utah_teapot', 'water_bottle', 'wine_glass'
split_participants = {
  'train': {
    'exclude_sessions': [5, 15, 25, 35, 45],
  },
  'test': {
    'include_sessions': [5, 15, 25, 35, 45]
  }
}  #]

split_overfit = {
  'train': {
    'include_sessions': [50],
    'include_instructions': ['use'],
    'include_objects': ['binoculars', 'camera'],
  },
  'test': {
    'include_sessions': [50],
    'include_instructions': ['use'],
    'include_objects': ['camera'],
  }
}
