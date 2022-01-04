import os
import io
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract
import cv2
import json
import re
import configparser
# from logging_module import getlogger
from scipy.ndimage.filters import rank_filter
from scipy.ndimage import interpolation as inter
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.transform import rotate
# LOG_DIR = "logs"
# current_dir_path = os.path.dirname(os.path.abspath(__file__))
# new_path = (os.path.abspath(os.path.join(current_dir_path, os.pardir)))
config_file_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Config.cfg")
config_obj = configparser.ConfigParser()

try:
    config_obj.read(config_file_loc)
    debug_level = int(config_obj.get("Logs", "debuglevel"))
    log_filename_config = config_obj.get("Logs", "logfilename")
    # tesseract_path = config_obj.get("Tesseract","tesseract_path")
    info_retain_threshold = float(config_obj.get("Tesseract","info_retain_threshold")) # default 0.005
except Exception as e:
    raise Exception("Config file error: " + str(e))

# log_filename = LOG_DIR + "/logs/" + log_filename_config
# logger_obj = getlogger.GetLogger(name="IMAGE_TEXT_EXTRACTOR", logfileloc=log_filename, debuglevel=debug_level)
# logger = logger_obj.getlogger1()

# pytesseract.pytesseract.tesseract_cmd = tesseract_path

class ImageTextExtractor:
    def __init__(self,sigma=3.0,num_peaks=20,r_angle=0):
        # try:
        self.sigma = sigma
        self.num_peaks = num_peaks
        self.r_angle = r_angle
        self.piby4 = np.pi / 4
        # except Exception as e:
        #     # logger.error("Error while initialising ImageTextExtractor : "+str(e))
        #     raise Exception("Error while initialising ImageTextExtractor : "+str(e))

    def downscale_image(self, im, max_dim=2048):
        """Shrink im until its longest dimension is <= max_dim.
        Returns new_image, scale (where scale <= 1).
        """
        # try:
        width, height = im.size
        if max(width, height) <= max_dim:
            return 1.0, im

        scale = 1.0 * max_dim / max(width, height)
        new_im = im.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
        return scale, new_im
        # except Exception as e:
        #     # logger.error("Error while downscaling image : "+str(e))
        #     return 1.0, im
        #     # raise Exception("Error while downscaling image : "+str(e))

    def find_border_components(self, contours, ary):
        # try:
        borders = []
        area = ary.shape[0] * ary.shape[1]
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 0.9 * area:
                borders.append((i, x, y, x + w - 1, y + h - 1))
        return borders
        # except Exception as e:
        #     # logger.error("Error while finding border components of an image : "+str(e))
        #     return []
            # raise Exception("Error while finding border components of an image : "+str(e))

    def angle_from_right(self, deg):
        # try:
        return min(deg % 90, 90 - (deg % 90))
        # except Exception as e:
        #     # logger.error("Error while finding angle of image from right : "+str(e))
        #     return deg
        #     # raise Exception("Error while finding angle of image from right : "+str(e))

    def remove_border(self, contour, ary):
        """Remove everything outside a border contour."""
        # Use a rotated rectangle (should be a good approximation of a border).
        # If it's far from a right angle, it's probably two sides of a border and
        # we should use the bounding box instead.
        # try:
        c_im = np.zeros(ary.shape)
        r = cv2.minAreaRect(contour)
        degs = r[2]
        if self.angle_from_right(degs) <= 10.0:
            box = cv2.boxPoints(r)
            box = np.int0(box)
            cv2.drawContours(c_im, [box], 0, 255, -1)
            cv2.drawContours(c_im, [box], 0, 0, 4)
        else:
            x1, y1, x2, y2 = cv2.boundingRect(contour)
            cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

        return np.minimum(c_im, ary)
        # except Exception as e:
        #     # logger.error("Error while removing border components of an image : "+str(e))
        #     return ary
            # raise Exception("Error while removing border components of an image : "+str(e))

    def dilate(self, ary, N, iterations):
        """Dilate using an NxN '+' sign shape. ary is np.uint8."""
        # try:
        kernel = np.zeros((N, N), dtype=np.uint8)
        kernel[(N - 1) // 2, :] = 1
        dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

        kernel = np.zeros((N, N), dtype=np.uint8)
        kernel[:, (N - 1) // 2] = 1
        dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
        return dilated_image
        # except Exception as e:
        #     # logger.error("Error while dilating image : "+str(e))
        #     return ary
            # raise Exception("Error while dilating image : "+str(e))

    def find_components(self, edges, max_components=32):
        """Dilate the image until there are just a few connected components.
        Returns contours for these components."""
        # Perform increasingly aggressive dilation until there are just a few
        # connected components.
        # try:
        total = np.sum(edges) / 255
        area = edges.shape[0] * edges.shape[1]
        if (total / area) < 0.005:
            max_components = 16
        dilation = 5
        count = max_components + dilation
        n = 1
        while count > max_components:
            n += 1
            dilated_image = self.dilate(edges, N=3, iterations=n)
            dilated_image = np.uint8(dilated_image)
            contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = len(contours)
        return contours
        # except Exception as e:
        #     # logger.error("Error while finding components (contours) of image : "+str(e))
        #     return []
            # raise Exception("Error while finding components (contours) of image : "+str(e))

    def props_for_contours(self, contours, ary):
        """Calculate bounding box & the number of set pixels for each contour."""
        # try:
        c_info = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            c_im = np.zeros(ary.shape)
            cv2.drawContours(c_im, [c], 0, 255, -1)
            c_info.append({
                "x1": x,
                "y1": y,
                "x2": x + w - 1,
                "y2": y + h - 1,
                "sum": np.sum(ary * (c_im > 0)) / 255
            })
        return c_info
        # except Exception as e:
        #     # logger.error("Error while props of contours of image : "+str(e))
        #     return []
            # raise Exception("Error while props of contours of image : "+str(e))

    def crop_area(self, crop):
        # try:
            return max(0, crop["x2"] - crop["x1"]) * max(0, crop["y2"] - crop["y1"])
        # except Exception as e:
        #     # logger.error("Error while finding area of crop",crop," : "+str(e))
        #     return 0
        #     # raise Exception("Error while finding area of crop",crop," : "+str(e))

    def find_optimal_subsets(self, contours, edges):
        """Find a crop which strikes a good balance of coverage/compactness.
                Returns an (x1, y1, x2, y2) tuple.
                """
        # try:
        c_info = self.props_for_contours(contours, edges)
        total = np.sum(edges) / 255
        area = edges.shape[0] * edges.shape[1]
        new_c_info = []
        for crop in c_info:
            recall = 1.0 * crop["sum"] / total
            prec = 1 - 1.0 * self.crop_area(crop) / area
            f1 = 2 * (prec * recall / (prec + recall))
            if f1>info_retain_threshold:
                new_c_info.append(crop)
        c_info = new_c_info
        c_info.sort(key=lambda x: x["x1"])
        c_info.sort(key=lambda x: x["y1"])

        new_c_info = []
        for c in c_info:
            new_crop = c
            # if self.check_if_exists_in_array(c, new_c_info):
            #     continue
            # else:
            new_c_info.append(new_crop)
        new_c_info = [(c["x1"],c["y1"],c["x2"],c["y2"]) for c in new_c_info]
        return new_c_info
        # except Exception as e:
        #     # logger.error("Error while finding optimal subsets of image : "+str(e))
        #     return []
            # raise Exception("Error while finding optimal subsets of image : "+str(e))

    def check_if_exists_in_array(self,incoming_crop,crops_list):
        # try:
            if incoming_crop in crops_list:
                return True
            does_exist = False
            for crop in crops_list:
                if incoming_crop["x1"]>=crop["x1"] and incoming_crop["x2"]<=crop["x2"]:
                    if incoming_crop["y1"] >= crop["y1"] and incoming_crop["y2"] <= crop["y2"]:
                        does_exist = True
                        break
            return does_exist
        # except Exception as e:
        #     # logger.error("Error while checking duplicacy of subsets of image : "+str(e))
        #     return False
            # raise Exception("Error while checking duplicacy of subsets of image : "+str(e))

    def remove_shadows(self,img,factor=0.5):
        # try:
            img = np.asarray(img)
            if len(img.shape) == 3:
                img = img[:, :, ::-1]
                mode = "RGB"
            else:
                img = img[:, :-1]
                mode = "L"
            rgb_planes = cv2.split(img)
            result_norm_planes = []
            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_norm_planes.append(norm_img)
            result_norm = cv2.merge(result_norm_planes)
            img_without_shadows = Image.fromarray(np.uint8(result_norm), mode)
            enhancer_object = ImageEnhance.Sharpness(img_without_shadows)
            out = enhancer_object.enhance(factor)
            return out

    def find_score(self,arr, angle):
        # try:
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score

    def deskew_partial(self,img,delta=1,limit=5):
        # try:
            wd, ht = img.size
            pix = np.array(img.convert('1').getdata(), np.uint8)
            bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
            angles = np.arange(-limit, limit + delta, delta)
            scores = []
            for angle in angles:
                hist, score = self.find_score(bin_img, angle)
                scores.append(score)
            best_score = max(scores)
            best_angle = angles[scores.index(best_score)]
            bg = Image.new("RGBA", img.size, (255,) * 4)
            im = img.convert("RGBA").rotate(best_angle, expand=True)
            bg.paste(im, im)
            return bg

    def get_max_freq_elem(self, arr):
        # try:
            max_arr = []
            freqs = {}
            for i in arr:
                if i in freqs:
                    freqs[i] += 1
                else:
                    freqs[i] = 1
            sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
            max_freq = freqs[sorted_keys[0]]
            for k in sorted_keys:
                if freqs[k] == max_freq:
                    max_arr.append(k)
            return max_arr

    def compare_sum(self, value):
        # try:
            if value >= 44 and value <= 46:
                return True
            else:
                return False

    def calculate_deviation(self, angle):
            angle_in_degrees = np.abs(angle)
            deviation = np.abs(self.piby4 - angle_in_degrees)
            return deviation

    def determine_skew(self, img):
        # try:
            edges = canny(img, sigma=self.sigma)
            h, a, d = hough_line(edges)
            _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)
            if len(ap) == 0:
                # logger.warning("Message: Bad Quality Image")
                return {"Message": "Bad Quality Image"}
            absolute_deviations = [self.calculate_deviation(k) for k in ap]
            average_deviation = np.mean(np.rad2deg(absolute_deviations))
            ap_deg = [np.rad2deg(x) for x in ap]
            bin_0_45 = []
            bin_45_90 = []
            bin_0_45n = []
            bin_45_90n = []
            for ang in ap_deg:
                deviation_sum = int(90 - ang + average_deviation)
                if self.compare_sum(deviation_sum):
                    bin_45_90.append(ang)
                    continue
                deviation_sum = int(ang + average_deviation)
                if self.compare_sum(deviation_sum):
                    bin_0_45.append(ang)
                    continue
                deviation_sum = int(-ang + average_deviation)
                if self.compare_sum(deviation_sum):
                    bin_0_45n.append(ang)
                    continue
                deviation_sum = int(90 + ang + average_deviation)
                if self.compare_sum(deviation_sum):
                    bin_45_90n.append(ang)
            angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
            lmax = 0
            for j in range(len(angles)):
                l = len(angles[j])
                if l > lmax:
                    lmax = l
                    maxi = j
            if lmax:
                ans_arr = self.get_max_freq_elem(angles[maxi])
                ans_res = np.mean(ans_arr)
            else:
                ans_arr = self.get_max_freq_elem(ap_deg)
                ans_res = np.mean(ans_arr)
            data = {"avg Deviation from pi/4": average_deviation,"Estimated Angle": ans_res,"Angle bins": angles}
            return data

    def deskew(self,img):
        # try:
            res = self.determine_skew(img)
            # try:
            angle = res['Estimated Angle']
            if angle >= 0 and angle <= 90:
                rot_angle = angle - 90 + self.r_angle
            if angle >= -45 and angle < 0:
                rot_angle = angle - 90 + self.r_angle
            if angle >= -90 and angle < -45:
                rot_angle = 90 + angle + self.r_angle
            rotated = rotate(img, rot_angle, resize=True)
            return rotated

    def remove_transparency(self,im, bg_colour=(255, 255, 255)):

        # Only process if image has transparency (http://stackoverflow.com/a/1963146)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = im.convert('RGBA').split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            bg = Image.new("RGBA", im.size, bg_colour + (255,))
            bg.paste(im, mask=alpha)
            return bg

        else:
            return im
    def process_image(self, byte_image):
            orig_im = byte_image
            greyScale_image = orig_im.convert('L')
            array_image = np.asarray(greyScale_image)
            array_image = array_image / 255

            deskewed_img = self.deskew(array_image)

            if deskewed_img is None:
                deskewed_img = array_image
            after_deskew_img = Image.fromarray(np.uint8(deskewed_img * 255), 'L')
            scale, im = self.downscale_image(after_deskew_img)

            im = self.remove_shadows(im)

            im = self.deskew_partial(im)
            if not os.path.isdir(os.path.join(os.getcwd(),"output")):
                os.mkdir(os.path.join(os.getcwd(),"output"))
            print(im.mode)
            # im = im[:, :, :3]
            # overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
            # im = Image.alpha_composite(im, overlay)
            rgb_im = im.convert('RGB')
            rgb_im.save('audacious.jpg')
            rgb_im.save(os.path.join(os.getcwd(),"output","res.jpg"),'JPEG', quality=80)




if __name__ == "__main__":
    obj = ImageTextExtractor()
    import time
    img = Image.open(r"/Users/prasingh/Prashant/Prashant/CareerBuilder/Extraction/data/Batch1/0064O00000j8HZKQA2-00P4O00001KCPC3UAP-Donna VC.jpg")
    st_time = time.time()
    text_seg = obj.process_image(img)
    print(text_seg)
    # print("Time Taken =====>", time.time() - st_time)
