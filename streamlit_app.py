"""
TACTIC Pointing Correction - Streamlit Web Version
Complete version with all features and proper credits
Developed by: Mahendra Kothari and Muskan Maheshwari
"""

import streamlit as st
import os
import numpy as np
from io import BytesIO
import tempfile
from TACTIC_Pointing_Correction import TACTICPointingCorrection

st.set_page_config(
    page_title="TACTIC Pointing Correction",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        color: white;
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .developer-credits {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 3px 5px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #434343 0%, #000000 100%);
        color: white;
        border-radius: 10px;
        margin-top: 40px;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: none;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔭 TACTIC Pointing Correction Tool</div>', unsafe_allow_html=True)

# Developer Credits
st.markdown(
    '<div class="developer-credits">'
    '👨‍💻 Developed by: Mahendra Kothari & Muskan Maheshwari 👩‍💻'
    '</div>',
    unsafe_allow_html=True
)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Source Configuration
st.sidebar.subheader("🎯 Source Details")
source_name = st.sidebar.text_input("Source Name", "MU_Geminorum")
source_ra = st.sidebar.number_input("RA (hours)", value=5.627411, format="%.6f")
source_dec = st.sidebar.number_input("DEC (degrees)", value=21.142584, format="%.6f")

# Observatory Configuration
st.sidebar.subheader("🌍 Observatory")
obs_lat = st.sidebar.number_input("Latitude", value=24.6548, format="%.4f")
obs_lon = st.sidebar.number_input("Longitude", value=72.7792, format="%.4f")
obs_height = st.sidebar.number_input("Height (m)", value=1360)

# File Upload Section
st.header("📁 Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.subheader("FITS Files")
    fits_files = st.file_uploader(
        "Upload FITS files (.fit)", 
        accept_multiple_files=True,
        type=['fit', 'fits'],
        help="Select all FITS files from your observation session"
    )
    if fits_files:
        st.success(f"✅ {len(fits_files)} FITS files uploaded")
    
with col2:
    st.subheader("Timestamp File")
    timestamp_file = st.file_uploader(
        "Upload timestamp file (.txt)",
        type=['txt'],
        help="File containing observation timestamps in IST"
    )
    if timestamp_file:
        st.success("✅ Timestamp file uploaded")

# Reference Points
st.header("🎯 Reference Points")

st.markdown(
    '<div class="info-box">'
    '<b>💡 Tip:</b> Enter LED coordinates (same for all frames) and Star coordinates. '
    'Set Post-Transit to 0,0 for auto-detection.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("LED Coordinates")
    led_a_x = st.number_input("LED A - X", value=546.0919, format="%.4f", key="led_a_x")
    led_a_y = st.number_input("LED A - Y", value=546.2835, format="%.4f", key="led_a_y")
    led_b_x = st.number_input("LED B - X", value=543.5715, format="%.4f", key="led_b_x")
    led_b_y = st.number_input("LED B - Y", value=119.8698, format="%.4f", key="led_b_y")

with col2:
    st.write("")  # Spacing
    st.write("")
    led_c_x = st.number_input("LED C - X", value=332.5692, format="%.4f", key="led_c_x")
    led_c_y = st.number_input("LED C - Y", value=336.5287, format="%.4f", key="led_c_y")
    led_d_x = st.number_input("LED D - X", value=755.4114, format="%.4f", key="led_d_x")
    led_d_y = st.number_input("LED D - Y", value=335.6598, format="%.4f", key="led_d_y")

with col3:
    st.subheader("Star Coordinates")
    star_pre_x = st.number_input("Star Pre-Transit - X", value=554.2152, format="%.4f", key="star_pre_x")
    star_pre_y = st.number_input("Star Pre-Transit - Y", value=301.6761, format="%.4f", key="star_pre_y")
    star_post_x = st.number_input("Star Post-Transit - X", value=0.0, format="%.4f", key="star_post_x", 
                                   help="Set to 0 for auto-detection")
    star_post_y = st.number_input("Star Post-Transit - Y", value=0.0, format="%.4f", key="star_post_y",
                                   help="Set to 0 for auto-detection")

# Polynomial Fitting Options
st.header("📊 Polynomial Fitting Options")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pre-Transit Orders")
    pre_orders = st.multiselect(
        "Select polynomial orders for Pre-Transit fitting",
        [1, 2, 3, 4, 5],
        default=[1, 2, 3],
        key="pre_orders"
    )

with col2:
    st.subheader("Post-Transit Orders")
    post_orders = st.multiselect(
        "Select polynomial orders for Post-Transit fitting",
        [1, 2, 3, 4, 5],
        default=[1, 2, 3],
        key="post_orders"
    )

# Analysis Button
st.markdown("---")
st.header("🚀 Run Analysis")

if st.button("▶️ START ANALYSIS", type="primary", use_container_width=True):
    if not fits_files or not timestamp_file:
        st.error("❌ Please upload FITS files and timestamp file!")
    else:
        with st.spinner("🔄 Processing... This may take a few minutes..."):
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save uploaded FITS files
                fits_dir = os.path.join(tmpdir, "fits")
                os.makedirs(fits_dir, exist_ok=True)
                
                for fits_file in fits_files:
                    with open(os.path.join(fits_dir, fits_file.name), 'wb') as f:
                        f.write(fits_file.read())
                
                # Save timestamp file
                timestamp_path = os.path.join(tmpdir, "timestamps.txt")
                with open(timestamp_path, 'wb') as f:
                    f.write(timestamp_file.read())
                
                # Initialize analyzer
                analyzer = TACTICPointingCorrection()
                analyzer.SOURCE_NAME = source_name
                analyzer.SOURCE_RA = source_ra * 15
                analyzer.SOURCE_DEC = source_dec
                analyzer.OBSERVATORY_LAT = obs_lat
                analyzer.OBSERVATORY_LON = obs_lon
                analyzer.OBSERVATORY_HEIGHT = obs_height
                analyzer.fits_dir = fits_dir
                analyzer.timestamp_file = timestamp_path
                analyzer.output_dir = tmpdir
                
                # Set reference points
                analyzer.reference_points = {
                    'A_up': (led_a_x, led_a_y),
                    'B_down': (led_b_x, led_b_y),
                    'C_left': (led_c_x, led_c_y),
                    'D_right': (led_d_x, led_d_y),
                    'Star_Pre': (star_pre_x, star_pre_y),
                    'Star_Post': (star_post_x, star_post_y),
                    'Star': (star_pre_x, star_pre_y)
                }
                
                try:
                    # Run analysis with progress updates
                    st.info("📊 Step 1/4: Parsing timestamps (IST → UTC)...")
                    timestamps = analyzer.parse_timestamps()
                    st.success(f"✓ Parsed {len(timestamps)} timestamps")
                    
                    st.info("🔍 Step 2/4: Processing FITS files and detecting objects...")
                    df = analyzer.process_fits_files(timestamps)
                    st.success(f"✓ Processed {len(df)} images")
                    
                    st.info("📈 Step 3/4: Creating polynomial fit plots...")
                    analyzer.plot_fits(df, pre_orders=pre_orders, post_orders=post_orders)
                    st.success("✓ Plots generated")
                    
                    st.info("💾 Step 4/4: Preparing download files...")
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Results Section
                    st.markdown("---")
                    st.header("📊 Results & Downloads")
                    
                    # Info about temporary storage
                    st.markdown(
                        '<div class="info-box">'
                        '<b>ℹ️ Storage & Download Information:</b><br>'
                        '• Files are processed in cloud temporary storage (not saved permanently)<br>'
                        '• Click download buttons below to save results to <b>your computer</b><br>'
                        '• Default save location: Your browser\'s Downloads folder<br>'
                        '&nbsp;&nbsp;→ Windows: <code>C:\\Users\\YourName\\Downloads</code><br>'
                        '&nbsp;&nbsp;→ Linux: <code>~/Downloads</code><br>'
                        '• To change location: Check your browser download settings or select "Ask where to save"'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display transit information
                    transit_idx = df['zenith'].idxmin()
                    transit_zenith = df.loc[transit_idx, 'zenith']
                    
                    st.subheader("🎯 Transit Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Transit Image", f"#{transit_idx + 1}")
                    with col2:
                        st.metric("Zenith Angle", f"{transit_zenith:.2f}°")
                    with col3:
                        st.metric("Total Images", len(df))
                    
                    # Show dataframe
                    st.subheader("📋 Data Table (First 20 rows)")
                    st.dataframe(df.head(20), use_container_width=True)
                    
                    st.info(f"💡 Full dataset has {len(df)} rows. Download Excel file for complete data.")
                    
                    # Download section with organized buttons
                    st.subheader("📥 Download Files")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Excel download
                    with col1:
                        excel_path = os.path.join(tmpdir, "pointing_correction_results.xlsx")
                        df.to_excel(excel_path, index=False)
                        
                        with open(excel_path, 'rb') as f:
                            st.download_button(
                                "📊 Download Excel Results",
                                f.read(),
                                "pointing_correction_results.xlsx",
                                "application/vnd.ms-excel",
                                use_container_width=True,
                                help="Complete data table with all measurements"
                            )
                    
                    # Plot download
                    with col2:
                        plot_path = os.path.join(tmpdir, "pointing_fits_pre_post.png")
                        if os.path.exists(plot_path):
                            with open(plot_path, 'rb') as f:
                                st.download_button(
                                    "📈 Download Plots",
                                    f.read(),
                                    "pointing_fits_pre_post.png",
                                    "image/png",
                                    use_container_width=True,
                                    help="Polynomial fit visualization"
                                )
                    
                    # Text results download
                    with col3:
                        txt_path = os.path.join(tmpdir, "fit_results_pre_post.txt")
                        if os.path.exists(txt_path):
                            with open(txt_path, 'rb') as f:
                                st.download_button(
                                    "📄 Download Fit Results",
                                    f.read(),
                                    "fit_results_pre_post.txt",
                                    "text/plain",
                                    use_container_width=True,
                                    help="Detailed fit statistics"
                                )
                    
                    # Show plots inline
                    st.subheader("📈 Pre/Post Transit Polynomial Fits")
                    if os.path.exists(plot_path):
                        st.image(plot_path, use_column_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Correction Statistics (X-axis):**")
                        st.write(f"- Mean: {df['Correction_X'].mean():.4f} arcmin")
                        st.write(f"- Std Dev: {df['Correction_X'].std():.4f} arcmin")
                        st.write(f"- Min: {df['Correction_X'].min():.4f} arcmin")
                        st.write(f"- Max: {df['Correction_X'].max():.4f} arcmin")
                    
                    with col2:
                        st.write("**Correction Statistics (Y-axis):**")
                        st.write(f"- Mean: {df['Correction_Y'].mean():.4f} arcmin")
                        st.write(f"- Std Dev: {df['Correction_Y'].std():.4f} arcmin")
                        st.write(f"- Min: {df['Correction_Y'].min():.4f} arcmin")
                        st.write(f"- Max: {df['Correction_Y'].max():.4f} arcmin")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    with st.expander("🔍 View Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                    st.info("💡 Tip: Check that your FITS files and timestamp file are properly formatted.")

# Information Section
st.markdown("---")
st.header("ℹ️ About This Tool")

with st.expander("📖 How to Use"):
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Upload Files:**
       - Select all FITS files from your observation
       - Upload the timestamp file (IST format)
    
    2. **Configure Settings (Sidebar):**
       - Set source coordinates (RA, Dec)
       - Set observatory location (Lat, Lon, Height)
    
    3. **Enter Reference Points:**
       - LED coordinates (A, B, C, D)
       - Star coordinates (Pre and Post transit)
       - Use 0,0 for Post-Transit if you want auto-detection
    
    4. **Select Polynomial Orders:**
       - Choose fitting orders for Pre and Post transit
       - Default: 1st, 2nd, 3rd order
    
    5. **Run Analysis:**
       - Click "START ANALYSIS" button
       - Wait for processing (2-5 minutes)
       - Download results
    
    ### Output Files:
    - **Excel:** Complete data table with all measurements
    - **Plots:** Polynomial fit visualizations
    - **Text:** Detailed fit statistics and coefficients
    """)

with st.expander("🔬 Technical Details"):
    st.markdown("""
    ### Analysis Method:
    - **LED Detection:** Advanced centroid detection with fallback to previous positions
    - **Star Detection:** DAOStarFinder algorithm with adaptive thresholding
    - **Transit Detection:** Automatic identification of minimum zenith angle
    - **Polynomial Fitting:** Separate fits for Pre and Post transit data
    - **Coordinate System:** IST timestamps converted to UTC for accuracy
    
    ### Calculations:
    - CCD Center = Average of 4 LED positions
    - Scale Factor = 16mm / (LED_B_y - LED_A_y)
    - Correction_X = (Star_X - Center_X) × Scale × 0.318 × 60 [arcmin]
    - Correction_Y = (Center_Y - Star_Y) × Scale × 0.318 × 60 [arcmin]
    """)

# Footer with credits
st.markdown(
    '<div class="footer">'
    '<h3>🔭 TACTIC Pointing Correction Tool</h3>'
    '<p style="font-size: 20px; margin: 15px 0;">'
    '👨‍💻 <b>Developed by:</b> Mahendra Kothari & Muskan Maheshwari 👩‍💻'
    '</p>'
    '<p style="margin: 10px 0;">Automated Telescope Pointing Analysis System</p>'
    '<p style="margin: 10px 0;">For astronomical observations and telescope calibration</p>'
    '<p style="margin-top: 20px; font-size: 12px; color: #aaa;">'
    'Version 2.0 | 2026 | Web-based Application'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)
