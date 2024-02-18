"""A few simple functions to run quantimus on a test image.

To run this test, install flika and install quantimus in the ~/.FLIKA/plugins 
directory. Then run this script from the command line. 

````bash
python tests/test_quantimus.py
```

It will open the test image and run multiple steps in the plugin pipeline.
"""

import pathlib
import flika


def generate_classification():
    flika_app = flika.start_flika()
    plugin_dir = pathlib.Path(flika.app.plugin_manager.get_plugin_directory())
    # Load the test image
    test_image_filename = plugin_dir / 'quantimus' / 'tests'/ 'laminin.tif'
    window = flika.process.file_.open_file(test_image_filename)
    # Select quantimus_plugin
    quantimus_plugin = flika.app.plugin_manager.PluginManager.plugins['QuantiMus']
    quantimus_plugin.menu.actions()[0].trigger() # Open the quantimus dialog

    quantimus = flika.global_vars.quantimus

    quantimus.original_window_selector.setWindow(window)
    quantimus.fill_boundaries_button()
    quantimus.binary_img_selector.setWindow(quantimus.binary_img)

    # Create positive and negative examples by clicking on the image.
    positives_coords = [(244, 363), (231, 351), (300, 410), (330, 316), (322, 327), (318, 337)]
    negative_coords = [(250, 375), (200, 354), (338, 333), (322, 360), (281, 381), (361, 478), (258, 428)]
    for coord in positives_coords:
        quantimus.classifier_window.mouse_left_click_inner(*coord)
    for coord in negative_coords:
        quantimus.classifier_window.mouse_left_click_inner(*coord)
        quantimus.classifier_window.mouse_left_click_inner(*coord)


    quantimus.run_svm_classification_on_image()
    return flika_app




if __name__ == '__main__':
    flika_app = generate_classification()
    flika_app.app.exec_()