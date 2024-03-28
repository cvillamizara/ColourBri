from matplotlib import pyplot as plt
import cv2
import pandas as pd
import numpy as np
from fastapi import FastAPI, Response, Form
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import File, UploadFile
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware


# uvicorn main:app --reload
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

debug: bool = False
COLOURS: list[list[float]]|None = None
SEGMENT_MASKS: list[list[bool]]|None = None


def get_image_from_path(pth):
    image = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
    if debug:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        plt.close()
    return image

def get_raw_contours(image):
    # converting image into grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        plt.imshow(image_grayscale, cmap = "gray")
        plt.axis('off')
        plt.show()
        plt.close()

    # making everything 1 or 0 depending on the threshold 
    _ , normal_mask = cv2.threshold(image_grayscale,240, 255, cv2.THRESH_BINARY_INV)

    normal_mask = cv2.erode(normal_mask, np.ones((7, 7), np.uint8))
    eroded_contours, _ = cv2.findContours(normal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        plt.imshow(normal_mask, cmap = "gray")
        plt.axis('off')
        cv2.imwrite('thresholding.png', cv2.hconcat([image, np.stack((normal_mask, normal_mask, normal_mask), axis=2)]))
        contours_img_before_filtering = normal_mask.copy()
        contours_img_before_filtering = cv2.cvtColor(contours_img_before_filtering, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contours_img_before_filtering, eroded_contours, -1, (0, 255, 0), 3)
        plt.imshow(contours_img_before_filtering)
        plt.axis('off')
        plt.show()
        plt.close()

    return eroded_contours, normal_mask

def mean_colours_from_raw_contours(image, contours, mask):
    filtered_contours = []
    df_segment_color = pd.DataFrame()
    for idx, contour in enumerate(contours):
        area = int(cv2.contourArea(contour))*3

        # if area is more than the 0.005% of the image :)
        if area > (0.005 * np.prod(image.shape))/100:
            filtered_contours.append(contour)
            masked = np.zeros_like(image[:, :, 0])  # This mask is used to get the mean color of the specific bead (contour)
            cv2.drawContours(masked, [contour], 0, 255, -1)
            [B_mean, G_mean, R_mean], [B_dev, G_dev, R_dev ] = cv2.meanStdDev(image, mask=masked)

            df = pd.DataFrame({'B_mean': B_mean, 'G_mean': G_mean, 'R_mean': R_mean, 'B_dev': B_dev, 'G_dev': G_dev, 'R_dev': R_dev}, index=[idx])
            df_segment_color = pd.concat([df_segment_color, df])
    
    if debug:
        contours_img_after_filtering = mask.copy()
        contours_img_after_filtering = cv2.cvtColor(contours_img_after_filtering, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contours_img_after_filtering, tuple(filtered_contours), -1, (0, 255, 0), 3)

        plt.imshow(contours_img_after_filtering)
        plt.axis('off')
        plt.show()
        plt.close()

    return df_segment_color

def assign_colour_idx(df_segment_color, tolerance=15):
    colours: list[list[int]] = []
    df_segment_color['colour_idx'] = None

    for row_index, row in df_segment_color.iterrows():
        b_row = row['B_mean']
        g_row = row['G_mean']
        r_row = row['R_mean']

        matched = False
        for idx in range(len(colours)):
            b, g, r  = colours[idx]
            if (pow(abs(b-b_row),2)+ pow(abs(g - g_row),2) + pow(abs(r - r_row),2)) < tolerance:
                df_segment_color.loc[row_index, 'colour_idx'] = idx
                matched = True
                break
            
        if not matched:
            df_segment_color.loc[row_index, 'colour_idx'] = len(colours)
            colours.append([b_row, g_row, r_row])
    return df_segment_color, colours

def get_colour_mask(image, contours, label_cnt_idx):
   mask = np.zeros_like(image[:, :, 0])
   cv2.drawContours(mask, [contours[i] for i in label_cnt_idx], -1, (255), -1)
   return mask

def group_colour_mask(image, df_segment_color, contours):
    segment_masks = []
    colours = [] # reusing the array to now actually have the average of the colour
    kernel = np.ones((7, 7), np.uint8)

    for _, df_grouped in df_segment_color.groupby('colour_idx'):
        avg_fcolour = df_grouped[['B_mean', 'G_mean', 'R_mean']].mean()
        # df_grouped are the indices to the figures within a same colour
        segment_masks.append(cv2.dilate(get_colour_mask(image, contours, df_grouped.index), kernel))
        colours.append([avg_fcolour["B_mean"], avg_fcolour["G_mean"], avg_fcolour["R_mean"]]) 

    return colours, segment_masks


def colours_in_image(colours):
    fig, ax = plt.subplots(1, len(colours), figsize=(len(colours), 1))

    for i, color in enumerate(colours):
        colour_int = [int(c) for c in reversed(color)]

        ax[i].imshow([[colour_int]])
        ax[i].axis('off')
        ax[i].set_title(str(i+1))
    
    plt.tight_layout()
    plt.savefig("colour_pallete.jpeg")

    if debug:
        plt.show()
        plt.close()

    # pallete = cv2.imread(r'colour_pallete.jpeg', cv2.IMREAD_UNCHANGED)
    # return pallete

    return "colour_pallete.jpeg"
    
def process_source(image):
    contours, bw_mask = get_raw_contours(image)
    df_segment_color = mean_colours_from_raw_contours(image, contours, bw_mask)
    df_segment_color, colours = assign_colour_idx(df_segment_color)
    colours, segment_masks = group_colour_mask(image, df_segment_color, contours)

    return colours, segment_masks



# def get_click_coordinates(row = 500, col = 1400):
def get_click_coordinates(row = 766, col = 766):
    return row, col

# def changing_mask(segment_masks):
#     row, col = get_click_coordinates()
#     for idx, m in enumerate(segment_masks):
#         if m[row][col]:
#             return idx, m 
#     return None

def changing_mask(segment_masks, idx):
    return segment_masks[idx-1]

def change_colour_group(image, new_colour, mask):
    img = image.copy()
    for r_idx, r in enumerate(img):
        for c_idx, c in enumerate(r):
            if mask[r_idx][c_idx]:
                img[r_idx][c_idx] = new_colour        

    return img

def change_colour(image, segment_masks, colours, new_colour, idx):
    mask_in_change = changing_mask(segment_masks, idx)
    if mask_in_change is None:
        print("I am sorry, you cant select this colour to be changed")
    new_image = change_colour_group(image, new_colour, mask_in_change)
    cv2.imwrite('copy.jpg', new_image)
    colours[idx - 1] = new_colour


def get_new_colour():
    return [128,0,128]


# @app.get("/1")
# async def root():
#     c_pallete = Path("colour_pallete.png")
#     return FileResponse(c_pallete)

# @app.get("/2")
# async def root():
#     final_pic = Path("color_segmentation_final.png")
#     return FileResponse(final_pic)


def process_colors(rgb_colors: list[list[float]]) -> list[dict]:
    processed_data = []
    for color in rgb_colors:
        hex_color = '#%02x%02x%02x' % (int(color[2]), int(color[1]), int(color[0]))
        processed_data.append({"color": hex_color})
    return processed_data

from PIL import Image
@app.post("/process_image")
async def process_image(file: UploadFile):
    image = file
    global COLOURS, SEGMENT_MASKS
    input_contents = await image.read()
    source_path = "source.jpg"
    copy_path = "copy.jpg"
        
    if image.filename.endswith(".png"):
        with open("source.png", 'wb') as imagefile:
            imagefile.write(input_contents)

        im = Image.open("source.png").convert('RGB') # open file as an image object
        im.save(source_path, "JPEG")
        im.save(copy_path, "JPEG")

    elif image.filename.endswith(".jpeg"):
        with open(source_path, 'wb') as imagefile:
            imagefile.write(input_contents)
        with open(copy_path, 'wb') as imagefile:
            imagefile.write(input_contents)

    source = get_image_from_path(source_path)
    COLOURS, SEGMENT_MASKS = process_source(source)
    # pallete_path = colours_in_image(COLOURS)    
    # with open(pallete_path,'rb') as textfile:
    #     output_contents = textfile.read()
    
    # # print(SEGMENT_MASKS, COLOURS)
    # return Response(content=output_contents, media_type="image/jpeg")
    processed_data = process_colors(COLOURS)
    return {"buttons": processed_data}


@app.post("/fetch_current_colors/")
async def create_buttons():
    global COLOURS
    processed_data = process_colors(COLOURS)
    return {"buttons": processed_data}

from PIL import ImageColor
@app.post("/change_try/")
async def change_image(change: Annotated[int, Form()], new_colour_hex: Annotated[str, Form()]):
    print(new_colour_hex)
    rgb_colour = list(ImageColor.getcolor(str(new_colour_hex), "RGB"))
    print(rgb_colour)
    

@app.post("/change/")
async def change_image(change: Annotated[int, Form()], new_colour_hex: Annotated[str, Form()]):
    new_colour_rgb = list(ImageColor.getcolor(new_colour_hex, "RGB"))
    new_colour = [i for i in reversed(new_colour_rgb)]
    global COLOURS, SEGMENT_MASKS
    idx = change
    # print(SEGMENT_MASKS, COLOURS)
    image_path = "copy.jpg"
    source = get_image_from_path(image_path)

    change_colour(source, SEGMENT_MASKS, COLOURS, new_colour, idx)

    with open(image_path,'rb') as textfile:
        output_contents = textfile.read()

    return Response(content=output_contents, media_type="image/jpeg")

@app.get("/reset")
async def reset_img():
    source_path = "source.jpg"
    copy_path = "copy.jpg"
    with open(source_path, 'rb') as src:
        contents = src.read()
    with open(copy_path, 'wb') as cpy:
        cpy.write(contents)


@app.post("/monochrome")
async def make_monochrome():
    global COLOURS, SEGMENT_MASKS
    n = len(COLOURS)
    # Create a new palette of colours from lighter to darker shades of grey
    new_colours = [[(i + 1) * 256/(n+1)] * 3 for i in range(n + 1)]
    
    # creating a translation map from the avg colours in the COLOURS array
    # this will make lighter colours be translated to lighter shades of grey
    colour_to_pos = {}
    for idx, c in enumerate(COLOURS):
        colour_to_pos[np.average(c)] = idx
    
    colour_to_pos = dict(sorted(colour_to_pos.items()))
    # print(SEGMENT_MASKS, COLOURS)
    image_path = "copy.jpg"
    
    i = 0
    for _, idx in colour_to_pos.items():
        source = get_image_from_path(image_path)
        new_colour = new_colours[i]
        i+=1
        change_colour(source, SEGMENT_MASKS, COLOURS, new_colour, idx)

    with open(image_path,'rb') as textfile:
        output_contents = textfile.read()

    return Response(content=output_contents, media_type="image/jpeg")