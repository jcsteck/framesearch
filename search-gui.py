"""
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import remi.gui as gui
import os
import operator
import time
import datetime
import warnings
import cv2
from remi import start, App
from threading import Timer
from sys import stdout


from skimage import color
from skimage import measure



class MyApp(App):
#record start time
start = time.process_time()

#ignore non-contiguous skimage warning
warnings.filterwarnings("ignore", module="skimage")


def prepare_image(filename):
    #open still image as rgb
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    #shrink
    img = cv2.resize(img, (10, 10))
    #convert to b&w
    img = color.rgb2gray(img)
    return img


def best_match(similarities):
    d = max(similarities, key=lambda x:x['similarity'])
    best_frame_number = d['frame']
    best_similarity = d['similarity']
    return best_frame_number, best_similarity


def parse_video(image, video, n_matches, break_point=False, verbose=False):
    #iterate through video frames
    
    similarities = [{'frame': 0, 'similarity': 0}]
    frame_count = 0
    
    #get current time
    fps_time = time.process_time()

    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):

        ret, frame = cap.read()

        #break at EOF
        if (type(frame) == type(None)):
            break

        #increment frame counter
        frame_count += 1

        #resize current video frame
        small_frame = cv2.resize(frame, (10, 10))
        #convert to greyscale
        small_frame_bw = color.rgb2gray(small_frame)

        #compare current frame to source image
        similarity = measure.compare_ssim(image, small_frame_bw)

        #remember current frame details
        similarities.append({'frame'      : frame_count,
                             'similarity' : similarity,
                             'image'      : frame})

        #find best match overall
        best_frame_number, best_similarity = best_match(similarities)
        
        #sort similarities list
        similarities = sorted(similarities, key=operator.itemgetter('similarity'), reverse=True)
        #remove surplus entries
        similarities = similarities[:n_matches]

        #calculate fps
        fps = frame_count / (time.process_time() - fps_time)

        #feedback to cli
        stdout.write('\r@ %d [%sfps] | best: %d (%s)  \r'
            % (frame_count, int(round(fps)), best_frame_number, round(best_similarity, 4), ))
        stdout.flush()

        #handle break_point
        if break_point:
            if similarity >= break_point:
                return similarities

    cap.release()
    return similarities


def sort_results(results, output=False):
    #sort results
    print('\n')
    sorted_results = sorted(results, key=operator.itemgetter('similarity'), reverse=True)
    n = 0
    print('\n--results:')
    for res in sorted_results:
        n += 1
        print('#%s\t%s\t%s\t: %s' % (n, res['filename'], res['frame'], res['similarity']))

        #save matched frames
        if output:
            save_frame(output, n, res['image'])


def save_frame(filename, n, image):
    fn, ext = filename.split('.')
    fn = '%s_%s.%s' % (fn, n, ext)
    cv2.imwrite(fn, image)


def walk(source_image, directory, number=1, break_point=False):
    results = []
    extentions = ['mp4', 'avi', 'mov', 'mkv', 'm4v']
    for root, dirs, files in os.walk(directory):
        for file in files:
            for ext in extentions:
                if file.endswith(ext):
                    video_fn = (os.path.join(root, file))
                    print(video_fn)
                    similarities = parse_video(source_image,
                                               video_fn,
                                               n_matches=number,
                                               break_point=break_point)
                     
                    #flatten results
                    for d in similarities:
                        results.append({'filename'   : video_fn,
                                        'frame'      : d['frame'],
                                        'similarity' : d['similarity'],
                                        'image'      : d['image']})

                        #stop walk if break point achieved
                        if break_point:
                            if d['similarity'] >= break_point:
                                return results

    return results


def main():
    import argparse

    #define cli arguments
    parser = argparse.ArgumentParser(description='''
        Compare an image with every frame of a video
        to find the best match.

        ============================================
        Edward Anderson
        --------------------------------------------
        v0.1 | 2016
        ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--image', help='source image')
    parser.add_argument('-v', '--video', help='video to search inside')
    parser.add_argument('-n', '--number', help='number of best matches to return', type=int, default=1)
    parser.add_argument('-b', '--break_point', help='stop searching when frame with [break_point] accuracy found; a number between 0 and 1', type=float, default=False)
    parser.add_argument('-o', '--output', help='filename.ext for best match; saved files are appended with "_n.ext"')
    parser.add_argument('-d', '--directory', help='directory of videos')
    args = parser.parse_args()

    #check source and destination provided
    if not args.image:
        parser.error('argument -i / --image is required')
    if not args.video:
        if not args.directory:
            parser.error('argument -v / --video is required')

    #prepare image
    source_image = prepare_image(args.image)

    #either walk directory or hande single file
    if args.directory:
        #scan directory and process each video file
        print('\n--reading videos:')
        results = walk(source_image, args.directory, args.number, args.break_point)
        s_results = sort_results(results, args.output)
        
    else:
        #process single video file
        print('\n--reading video:')
        similarities = parse_video(source_image,
                                   args.video,
                                   n_matches=args.number,
                                   break_point=args.break_point)

        print('\n\n--results:')
        #results to cli
        n = 0
        for d in similarities:
            n += 1
            print('#%s\t%s\t: %s' % (n, d['frame'], d['similarity']))
            
            #save matched frames
            if args.output:
                save_frame(args.output, n, d['image'])

    seconds_taken = time.process_time() - start
    time_taken = str(datetime.timedelta(seconds=seconds_taken))
    print('\n--time taken: \n%s\n' % time_taken)


if __name__ == '__main__':
    main()

    def __init__(self, *args):
        super(MyApp, self).__init__(*args)

    def idle(self):
        self.counter.set_text('Running Time: ' + str(self.count))
        self.progress.set_value(self.count%100)

    def main(self):
        # the margin 0px auto centers the main container
        verticalContainer = gui.Widget(width=540, margin='0px auto', style={'display': 'block', 'overflow': 'hidden'})

        horizontalContainer = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL, margin='0px', style={'display': 'block', 'overflow': 'auto'})
        
        subContainerLeft = gui.Widget(width=320, style={'display': 'block', 'overflow': 'auto', 'text-align': 'center'})
        self.img = gui.Image('/res:logo.png', height=100, margin='10px')
        self.img.onclick.do(self.on_img_clicked)

        self.table = gui.Table.new_from_list([('ID', 'First Name', 'Last Name'),
                                   ('101', 'Danny', 'Young'),
                                   ('102', 'Christine', 'Holand'),
                                   ('103', 'Lars', 'Gordon'),
                                   ('104', 'Roberto', 'Robitaille'),
                                   ('105', 'Maria', 'Papadopoulos')], width=300, height=200, margin='10px')
        self.table.on_table_row_click.do(self.on_table_row_click)

        # the arguments are	width - height - layoutOrientationOrizontal
        subContainerRight = gui.Widget(style={'width': '220px', 'display': 'block', 'overflow': 'auto', 'text-align': 'center'})
        self.count = 0
        self.counter = gui.Label('', width=200, height=30, margin='10px')

        self.lbl = gui.Label('This is a LABEL!', width=200, height=30, margin='10px')

        self.bt = gui.Button('Press me!', width=200, height=30, margin='10px')
        # setting the listener for the onclick event of the button
        self.bt.onclick.do(self.on_button_pressed)

        self.txt = gui.TextInput(width=200, height=30, margin='10px')
        self.txt.set_text('This is a TEXTAREA')
        self.txt.onchange.do(self.on_text_area_change)

        self.spin = gui.SpinBox(1, 0, 100, width=200, height=30, margin='10px')
        self.spin.onchange.do(self.on_spin_change)

        self.progress = gui.Progress(1, 100, width=200, height=5)

        self.check = gui.CheckBoxLabel('Label checkbox', True, width=200, height=30, margin='10px')
        self.check.onchange.do(self.on_check_change)

        self.btInputDiag = gui.Button('Open InputDialog', width=200, height=30, margin='10px')
        self.btInputDiag.onclick.do(self.open_input_dialog)

        self.btFileDiag = gui.Button('File Selection Dialog', width=200, height=30, margin='10px')
        self.btFileDiag.onclick.do(self.open_fileselection_dialog)

        self.btUploadFile = gui.FileUploader('./', width=200, height=30, margin='10px')
        self.btUploadFile.onsuccess.do(self.fileupload_on_success)
        self.btUploadFile.onfailed.do(self.fileupload_on_failed)

        items = ('Danny Young','Christine Holand','Lars Gordon','Roberto Robitaille')
        self.listView = gui.ListView.new_from_list(items, width=300, height=120, margin='10px')
        self.listView.onselection.do(self.list_view_on_selected)

        self.link = gui.Link("http://localhost:8081", "A link to here", width=200, height=30, margin='10px')

        self.dropDown = gui.DropDown.new_from_list(('DropDownItem 0', 'DropDownItem 1'),
                                                   width=200, height=20, margin='10px')
        self.dropDown.onchange.do(self.drop_down_changed)
        self.dropDown.select_by_value('DropDownItem 0')

        self.slider = gui.Slider(10, 0, 100, 5, width=200, height=20, margin='10px')
        self.slider.onchange.do(self.slider_changed)

        self.colorPicker = gui.ColorPicker('#ffbb00', width=200, height=20, margin='10px')
        self.colorPicker.onchange.do(self.color_picker_changed)

        self.date = gui.Date('2015-04-13', width=200, height=20, margin='10px')
        self.date.onchange.do(self.date_changed)

        self.video = gui.Widget( _type='iframe', width=290, height=200, margin='10px')
        self.video.attributes['src'] = "https://drive.google.com/file/d/0B0J9Lq_MRyn4UFRsblR3UTBZRHc/preview"
        self.video.attributes['width'] = '100%'
        self.video.attributes['height'] = '100%'
        self.video.attributes['controls'] = 'true'
        self.video.style['border'] = 'none'
                                     
        self.tree = gui.TreeView(width='100%', height=300)
        ti1 = gui.TreeItem("Item1")
        ti2 = gui.TreeItem("Item2")
        ti3 = gui.TreeItem("Item3")
        subti1 = gui.TreeItem("Sub Item1")
        subti2 = gui.TreeItem("Sub Item2")
        subti3 = gui.TreeItem("Sub Item3")
        subti4 = gui.TreeItem("Sub Item4")
        subsubti1 = gui.TreeItem("Sub Sub Item1")
        subsubti2 = gui.TreeItem("Sub Sub Item2")
        subsubti3 = gui.TreeItem("Sub Sub Item3")
        self.tree.append([ti1, ti2, ti3])
        ti2.append([subti1, subti2, subti3, subti4])
        subti4.append([subsubti1, subsubti2, subsubti3])
        
        # appending a widget to another, the first argument is a string key
        subContainerRight.append([self.counter, self.lbl, self.bt, self.txt, self.spin, self.progress, self.check, self.btInputDiag, self.btFileDiag])
        # use a defined key as we replace this widget later
        fdownloader = gui.FileDownloader('download test', '../remi/res/logo.png', width=200, height=30, margin='10px')
        subContainerRight.append(fdownloader, key='file_downloader')
        subContainerRight.append([self.btUploadFile])
        self.subContainerRight = subContainerRight

        subContainerLeft.append([self.img, self.table, self.listView, self.link, self.video])

        horizontalContainer.append([subContainerLeft, subContainerRight])

        menu = gui.Menu(width='100%', height='30px')
        m1 = gui.MenuItem('File', width=100, height=30)
        m2 = gui.MenuItem('View', width=100, height=30)
        m2.onclick.do(self.menu_view_clicked)
        m11 = gui.MenuItem('Save', width=100, height=30)
        m12 = gui.MenuItem('Open', width=100, height=30)
        m12.onclick.do(self.menu_open_clicked)
        m111 = gui.MenuItem('Save', width=100, height=30)
        m111.onclick.do(self.menu_save_clicked)
        m112 = gui.MenuItem('Save as', width=100, height=30)
        m112.onclick.do(self.menu_saveas_clicked)
        m3 = gui.MenuItem('Dialog', width=100, height=30)
        m3.onclick.do(self.menu_dialog_clicked)

        menu.append([m1, m2, m3])
        m1.append([m11, m12])
        m11.append([m111, m112])

        menubar = gui.MenuBar(width='100%', height='30px')
        menubar.append(menu)

        verticalContainer.append([menubar, horizontalContainer])

        #this flag will be used to stop the display_counter Timer
        self.stop_flag = False 

        # kick of regular display of counter
        self.display_counter()

        # returning the root widget
        return verticalContainer

    def display_counter(self):
        self.count += 1
        if not self.stop_flag:
            Timer(1, self.display_counter).start()

    def menu_dialog_clicked(self, widget):
        self.dialog = gui.GenericDialog(title='Dialog Box', message='Click Ok to transfer content to main page', width='500px')
        self.dtextinput = gui.TextInput(width=200, height=30)
        self.dtextinput.set_value('Initial Text')
        self.dialog.add_field_with_label('dtextinput', 'Text Input', self.dtextinput)

        self.dcheck = gui.CheckBox(False, width=200, height=30)
        self.dialog.add_field_with_label('dcheck', 'Label Checkbox', self.dcheck)
        values = ('Danny Young', 'Christine Holand', 'Lars Gordon', 'Roberto Robitaille')
        self.dlistView = gui.ListView.new_from_list(values, width=200, height=120)
        self.dialog.add_field_with_label('dlistView', 'Listview', self.dlistView)

        self.ddropdown = gui.DropDown.new_from_list(('DropDownItem 0', 'DropDownItem 1'),
                                                    width=200, height=20)
        self.dialog.add_field_with_label('ddropdown', 'Dropdown', self.ddropdown)

        self.dspinbox = gui.SpinBox(min=0, max=5000, width=200, height=20)
        self.dspinbox.set_value(50)
        self.dialog.add_field_with_label('dspinbox', 'Spinbox', self.dspinbox)

        self.dslider = gui.Slider(10, 0, 100, 5, width=200, height=20)
        self.dspinbox.set_value(50)
        self.dialog.add_field_with_label('dslider', 'Slider', self.dslider)

        self.dcolor = gui.ColorPicker(width=200, height=20)
        self.dcolor.set_value('#ffff00')
        self.dialog.add_field_with_label('dcolor', 'Colour Picker', self.dcolor)

        self.ddate = gui.Date(width=200, height=20)
        self.ddate.set_value('2000-01-01')
        self.dialog.add_field_with_label('ddate', 'Date', self.ddate)

        self.dialog.confirm_dialog.do(self.dialog_confirm)
        self.dialog.show(self)

    def dialog_confirm(self, widget):
        result = self.dialog.get_field('dtextinput').get_value()
        self.txt.set_value(result)

        result = self.dialog.get_field('dcheck').get_value()
        self.check.set_value(result)

        result = self.dialog.get_field('ddropdown').get_value()
        self.dropDown.select_by_value(result)

        result = self.dialog.get_field('dspinbox').get_value()
        self.spin.set_value(result)

        result = self.dialog.get_field('dslider').get_value()
        self.slider.set_value(result)

        result = self.dialog.get_field('dcolor').get_value()
        self.colorPicker.set_value(result)

        result = self.dialog.get_field('ddate').get_value()
        self.date.set_value(result)

        result = self.dialog.get_field('dlistView').get_value()
        self.listView.select_by_value(result)

    # listener function
    def on_img_clicked(self, widget):
        self.lbl.set_text('Image clicked!')

    def on_table_row_click(self, table, row, item):
        self.lbl.set_text('Table Item clicked: ' + item.get_text())

    def on_button_pressed(self, widget):
        self.lbl.set_text('Button pressed! ')
        self.bt.set_text('Hi!')

    def on_text_area_change(self, widget, newValue):
        self.lbl.set_text('Text Area value changed!')

    def on_spin_change(self, widget, newValue):
        self.lbl.set_text('SpinBox changed, new value: ' + str(newValue))

    def on_check_change(self, widget, newValue):
        self.lbl.set_text('CheckBox changed, new value: ' + str(newValue))

    def open_input_dialog(self, widget):
        self.inputDialog = gui.InputDialog('Input Dialog', 'Your name?',
                                           initial_value='type here', 
                                           width=500, height=160)
        self.inputDialog.confirm_value.do(
            self.on_input_dialog_confirm)

        # here is returned the Input Dialog widget, and it will be shown
        self.inputDialog.show(self)

    def on_input_dialog_confirm(self, widget, value):
        self.lbl.set_text('Hello ' + value)

    def open_fileselection_dialog(self, widget):
        self.fileselectionDialog = gui.FileSelectionDialog('File Selection Dialog', 'Select files and folders', False,
                                                           '.')
        self.fileselectionDialog.confirm_value.do(
            self.on_fileselection_dialog_confirm)

        # here is returned the Input Dialog widget, and it will be shown
        self.fileselectionDialog.show(self)

    def on_fileselection_dialog_confirm(self, widget, filelist):
        # a list() of filenames and folders is returned
        self.lbl.set_text('Selected files: %s' % ','.join(filelist))
        if len(filelist):
            filename = filelist[0]
            # replace the last download link
            fdownloader = gui.FileDownloader("download selected", f, width=200, height=30)
            self.subContainerRight.append(fdownloader, key='file_downloader')

    def list_view_on_selected(self, widget, selected_item_key):
        """ The selection event of the listView, returns a key of the clicked event.
            You can retrieve the item rapidly
        """
        self.lbl.set_text('List selection: ' + self.listView.children[selected_item_key].get_text())

    def drop_down_changed(self, widget, value):
        self.lbl.set_text('New Combo value: ' + value)

    def slider_changed(self, widget, value):
        self.lbl.set_text('New slider value: ' + str(value))

    def color_picker_changed(self, widget, value):
        self.lbl.set_text('New color value: ' + value)

    def date_changed(self, widget, value):
        self.lbl.set_text('New date value: ' + value)

    def menu_save_clicked(self, widget):
        self.lbl.set_text('Menu clicked: Save')

    def menu_saveas_clicked(self, widget):
        self.lbl.set_text('Menu clicked: Save As')

    def menu_open_clicked(self, widget):
        self.lbl.set_text('Menu clicked: Open')

    def menu_view_clicked(self, widget):
        self.lbl.set_text('Menu clicked: View')

    def fileupload_on_success(self, widget, filename):
        self.lbl.set_text('File upload success: ' + filename)

    def fileupload_on_failed(self, widget, filename):
        self.lbl.set_text('File upload failed: ' + filename)

    def on_close(self):
        """ Overloading App.on_close event to stop the Timer.
        """
        self.stop_flag = True
        super(MyApp, self).on_close()


if __name__ == "__main__":
    # starts the webserver
    # optional parameters
    # start(MyApp,address='127.0.0.1', port=8081, multiple_instance=False,enable_file_cache=True, update_interval=0.1, start_browser=True)
    import ssl
    start(MyApp, debug=True, address='0.0.0.0', port=8081, start_browser=True, multiple_instance=True)
