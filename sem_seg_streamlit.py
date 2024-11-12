import json
import os
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Semantic segmentation Data Analysis", layout="wide")

# Define color palette for labels
PALETTE = {
    "Radius": (220, 20, 60),          # Crimson Red
    "finger-1": (119, 11, 32),        # Dark Red
    "finger-2": (0, 0, 142),          # Dark Blue
    "finger-3": (0, 0, 230),          # Medium Blue
    "finger-4": (106, 0, 228),        # Purple
    "finger-5": (0, 60, 100),         # Dark Cyan
    "finger-6": (0, 80, 100),         # Deep Cyan
    "finger-7": (0, 0, 70),           # Very Dark Blue
    "finger-8": (0, 0, 192),          # Bright Blue
    "finger-9": (250, 170, 30),       # Orange Yellow
    "finger-10": (100, 170, 30),      # Olive Green
    "finger-11": (220, 220, 0),       # Yellow
    "finger-12": (175, 116, 175),     # Mauve
    "finger-13": (250, 0, 30),        # Bright Red
    "finger-14": (165, 42, 42),       # Brown
    "finger-15": (255, 77, 255),      # Pink
    "finger-16": (0, 226, 252),       # Cyan
    "finger-17": (182, 182, 255),     # Light Blue
    "finger-18": (0, 82, 0),          # Dark Green
    "finger-19": (120, 166, 157),     # Muted Teal
    "Trapezoid": (110, 76, 0),        # Dark Brown
    "Scaphoid": (174, 57, 255),       # Violet
    "Trapezium": (199, 100, 0),       # Reddish Orange
    "Lunate": (72, 0, 118),           # Deep Purple
    "Triquetrum": (255, 179, 240),    # Light Pink
    "Hamate": (0, 125, 92),           # Dark Teal
    "Capitate": (209, 0, 151),        # Magenta
    "Ulna": (188, 208, 182),          # Pale Green
    "Pisiform": (0, 220, 176)         # Bright Teal
}

@st.cache_data
def load_data():
    """
    Load train and test image paths along with corresponding label paths.
    """
    def load_image_paths(base_dir, has_labels=True):
        data = {'image_path': [], 'label_path': [] if has_labels else None}
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                data['image_path'].append(img_path)
                if has_labels:
                    label_path = os.path.join('train/outputs_json', folder, f'{img[:-4]}.json')
                    data['label_path'].append(label_path)
        return pd.DataFrame(data)
    
    train = load_image_paths('train/DCM')
    test = load_image_paths('test/DCM', has_labels=False)
    return train, test

@st.cache_data()
def rle_to_pixels(rle, shape):
    """
    Convert RLE data to (x, y) pixel coordinates.
    """
    pixels = []
    rle_pairs = [int(x) for x in rle.split()]
    for idx in range(0, len(rle_pairs), 2):
        start_pixel = rle_pairs[idx]
        run_length = rle_pairs[idx + 1]
        pixels.extend(range(start_pixel, start_pixel + run_length))
    return [(p % shape[1], p // shape[1]) for p in pixels]

@st.cache_data()
def test_labeled_image(img, image_path, csv):
    """
    Label the image using the CSV data.
    """
    if not csv.empty:
        csv = csv.fillna('')
        img_name = os.path.basename(image_path)
        labels = csv[csv['image_name'] == img_name]
        for _, row in labels.iterrows():
            pixels = rle_to_pixels(row['rle'], img.shape)
            for x, y in pixels:
                cv2.circle(img, (x, y), radius=1, color=PALETTE[row['class']], thickness=1)
    return img

def get_image(image_path, label_path=None, mode='train', csv=pd.DataFrame()):
    """
    Get the image with optional labeled annotations based on the mode.
    """
    img = cv2.imread(image_path)
    if st.session_state.get('show_label', False):
        if mode == 'train' and label_path:
            label_data = json.load(open(label_path))
            for annotation in label_data['annotations']:
                points = np.array(annotation['points'], dtype=np.int32)
                cv2.polylines(img, [points], True, PALETTE[annotation['label']], 10)
        elif mode == 'test':
            img = test_labeled_image(img, image_path, csv)
    return img

def show_images(image_paths, window, mode, csv=pd.DataFrame()):
    """
    Display images in a 2-column layout.
    """
    cols = window.columns(2)
    for idx, (path, anno) in enumerate(image_paths.values):
        if idx % 2 == 0:
            cols = window.columns(2)
        img = get_image(path, anno, mode, csv)
        cols[idx % 2].image(img)
        cols[idx % 2].write(f"Path: {path}")

def show_dataframe(data, window, mode, csv=pd.DataFrame()):
    """
    Display data in a paginated format with sorting options.
    """
    # Sorting and pagination settings
    top_menu = window.columns(3)
    with top_menu[0]:
        sort = st.radio(label = "Sort Data", options=["Yes", "No"], horizontal=True, index=1)
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options='image_path')
        with top_menu[2]:
            sort_direction = st.radio("Direction", options=["⬆️", "⬇️"], horizontal=True)
        data = data.sort_values(by=sort_field, ascending=(sort_direction == "⬆️"), ignore_index=True)

    # Page size and selection
    bottom_menu = window.columns((4, 1, 1))
    with bottom_menu[2]:
        page_size = st.selectbox("Page Size", options=[10, 20, 30])
    with bottom_menu[1]:
        total_pages = len(data) // page_size + (1 if len(data) % page_size > 0 else 0)
        current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}**")
    pages = [data.iloc[i:i + page_size] for i in range(0, len(data), page_size)] # split page

    # Display current page data
    con1, con2 = window.columns((1, 3))
    con1.dataframe(data=pages[current_page - 1]['image_path'], use_container_width=True)
    show_images(pages[current_page - 1], con2, mode, csv)

@st.dialog("CSV Upload")
def upload_csv(csv):
    """
    Upload and preview a CSV file.
    """
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df[['image_name', 'class', 'rle']]
        st.write("Data Preview:")
        st.dataframe(df)
        input_name = st.text_input("Name your CSV file", value=uploaded_file.name.replace('.csv', ''))
        if st.button("Upload CSV"):
            output_name = check_same_csv(input_name + '.csv', csv)
            st.write("Saved file name: " + output_name)
            df.to_csv('outputs/' + output_name, index=False)
        if st.button("Close"):
            st.rerun()

def check_same_csv(name, csv_files):
    """
    Ensure the CSV file name is unique by appending an incrementing number if necessary.
    """
    i = 1
    original_name = name
    while name in csv_files:
        name = f"{original_name[:-4]}_{i}.csv" if i == 1 else f"{original_name[:-6]}_{i}.csv"
        i += 1
    return name

def main():
    ## login
    if 'login' not in st.session_state or not st.session_state['login']:
        auth = {'7148', '7202', '7214', '7218', '7258', '7263'}
        password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.button('Login'):
            if password in auth:
                st.session_state['login'] = True
            else:
                st.write('Invalid password')
        return

    if st.sidebar.button("Refresh"):
        st.rerun()
    
    # Sidebar options
    option = st.sidebar.selectbox("Select Data Type", ("Image Data", "Ensemble Candidation"))
    train_data, test_data = load_data()

    if option == "Image Data":
        data_type = st.sidebar.selectbox("Train/Test", ("train", "test"))
        st.session_state['show_label'] = st.sidebar.checkbox("Show Labels", value=True)

        if data_type == "train":
            st.header("Train Data")
            show_dataframe(train_data, st, 'train')

        elif data_type == "test":
            st.header("Test Data")
            os.makedirs('outputs/', exist_ok=True)

            csv_files = [f for f in os.listdir('outputs') if f.endswith('.csv')]
            chosen_csv = st.sidebar.selectbox("Apply Output CSV", ("None",) + tuple(csv_files))

            current_csv = pd.read_csv('outputs/' + chosen_csv) if chosen_csv != "None" else pd.DataFrame()
            show_dataframe(test_data, st, 'test', current_csv)

            if st.sidebar.button("Upload New CSV"):
                upload_csv(csv_files)

if __name__ == "__main__":
    main()
