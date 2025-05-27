#ifndef	MY_DEFINITION
#define MY_DEFINITION

//#define MAC_OS
#define WIN
//#define LINUX

//widget names
#define GRPAH_WIDGET_NAME "Graph Display"
#define RENDER_WIDGET_NAME "Render Widget"
//messages
#define SELECT_LINE_CHANGE_MSG "Selected Streamline Change"
#define SELECT_SEGMENT_CHANGE_MSG "Selected Segment Change"
#define SELECT_POINT_CHANGE_MSG "Selected Point Change"
#define LINE_RADIUS_CHANGE_MSG "Line Radius Change"
#define SELECT_SUBJECT_CHANGE_MSG "subject Change"
#define SELECT_ROI_CHANGE_MSG "roi Change"
#define FILTER_LINES_MSG "filter streamlines change"


//global data items
#define SELECTED_LINE_ID_NAME "Global.Streamline"
#define SELECTED_POINT_ID_NAME "Global.Point ID"
//#define SELECTED_SUBJECT_INDEX "Global.Subject"

//COMPUTER_ROOM CRAYFISH HURRICANE LIFTED_SUB_SLAB LIFTED_SUB OUTCAR PLUME
//FIVE_CRITICAL SUPERNOVA_100 SUPERNOVA_864 TORNADO TWO_SWIRL ELECTRO3D
//VSFS9 VSFS9_LARGE SQUARE_CYLINDER ABC_FLOW HURRICANE_DOWNSAMPLE CLOUD CASE_119
//HURRICANE48 ATMOSPHERICAL OSU_DNS LES_JET GL3D GL3D2 
//ATMOSPHERICAL_UNSTEADY LES_JET_UNSTEADY OSU_DNS_UNSTEADY
#define PLUME

#define LOAD_STREAMLINE_POOL
#define DEFUALT_SIMPLFY_INTERVAL		1
#define DEFUALT_ENCODING_THRESHOLD		0.5f
#define RESAMPLE_THRESHOLD		1.0f

#define ADD_STREAMLINE_NUM      0
#define GENERATE_STREAMLINE_NUM 3000
#define MAX_STREAMLINE_NUM		3000
#define MAX_POINT_NUM			300
#define MIN_POINT_NUM			10
#define TRACE_INTERVAL			0.05f
#define MAX_STEP_NUM			5000
#define MAX_POINT_NUM_BINORMAL	300
#define SEG_LEN_BINORMAL		1.0f

#ifdef COMPUTER_ROOM
#define VECFIELD_NAME				"computer_room"
#define VECFIELD_PATH				"E:/data/flow/computer_room.vec"
#define VECHEADER_PATH				"E:/data/flow/computer_room.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.6f
#elif defined CRAYFISH
#define VECFIELD_NAME				"crayfish"
#define VECFIELD_PATH				"E:/data/flow/crayfish.vec"
#define VECHEADER_PATH				"E:/data/flow/crayfish.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.8f
#elif defined HURRICANE
#define VECFIELD_NAME				"hurricane"
#define VECFIELD_PATH				"E:/data/flow/hurricane.vec"
#define VECHEADER_PATH				"E:/data/flow/hurricane.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined HURRICANE48
#define VECFIELD_NAME				"hurricane"
#define VECFIELD_PATH				"E:/data/flow/hurricane_48.vec"
#define VECHEADER_PATH				"E:/data/flow/hurricane.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined HURRICANE_DOWNSAMPLE
#define VECFIELD_NAME				"hurricane_downsample"
#define VECFIELD_PATH				"E:/data/flow/hurricane_downsample.vec"
#define VECHEADER_PATH				"E:/data/flow/hurricane_downsample.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined LIFTED_SUB_SLAB
#define VECFIELD_NAME				"lifted_sub_slab"
#define VECFIELD_PATH				"E:/data/flow/lifted_sub_slab.vec"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined LIFTED_SUB
#define VECFIELD_NAME				"lifted_sub"
#define VECFIELD_PATH				"E:/data/flow/lifted_sub.vec"
#define VECHEADER_PATH				"E:/data/flow/lifted_sub.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined OUTCAR
#define VECFIELD_NAME				"outcar"
#define VECFIELD_PATH				"E:/data/flow/outcar.vec"
#define VECHEADER_PATH				"E:/data/flow/outcar.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined PLUME
#define VECFIELD_NAME				"plume"
#define VECFIELD_PATH				"E:/data/flow/plume.vec"
#define VECHEADER_PATH				"E:/data/flow/plume.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined FIVE_CRITICAL
#define VECFIELD_NAME				"5cp"
#define VECFIELD_PATH				"E:/data/flow/random-5cp.vec"
#define VECHEADER_PATH				"E:/data/flow/random-5cp.hdr"
#define	SEG_LEN						0.5f
#define TUBE_RADIUS					0.2f
#elif defined SUPERNOVA_100
#define VECFIELD_NAME				"supernova100"
#define VECFIELD_PATH				"E:/data/flow/supernova100.vec"
#define VECHEADER_PATH				"E:/data/flow/supernova100.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined SUPERNOVA_864
#define VECFIELD_NAME				"supernova864"
#define VECFIELD_PATH				"E:/data/flow/supernova864.vec"
#define VECHEADER_PATH				"E:/data/flow/supernova864.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined TORNADO
#define VECFIELD_NAME				"tornado"
#define VECFIELD_PATH				"E:/data/flow/tornado.vec"
#define VECHEADER_PATH				"E:/data/flow/tornado.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined TWO_SWIRL
#define VECFIELD_NAME				"two_swirl"
#define VECFIELD_PATH				"E:/data/flow/two_swirl.vec"
#define VECHEADER_PATH				"E:/data/flow/two_swirl.hdr"
#define	SEG_LEN						0.8f
#define TUBE_RADIUS					0.2f
#elif defined ELECTRO3D
#define VECFIELD_NAME				"electro3D"
#define VECFIELD_PATH				"E:/data/flow/electro3D.vec"
#define VECHEADER_PATH				"E:/data/flow/electro3D.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined CASE_119
#define VECFIELD_NAME				"case119"
#define VECFIELD_PATH				"E:/data/flow/case119_15.vec"
#define VECHEADER_PATH				"E:/data/flow/case119.hdr"
#define	SEG_LEN						1.0f
#define TUBE_RADIUS					0.2f
#elif defined VSFS9
#define VECFIELD_NAME				"vsfs9"
#define VECFIELD_PATH				"E:/data/flow/vsfs9_25.vec"
#define VECHEADER_PATH				"E:/data/flow/vsfs9.hdr"
#define	SEG_LEN						1.0f
#define TUBE_RADIUS					0.2f
#elif defined VSFS9_LARGE
#define VECFIELD_NAME				"vsfs9_large"
#define VECFIELD_PATH				"E:/data/flow/vsfs9_large.vec"
#define VECHEADER_PATH				"E:/data/flow/vsfs9_large.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined CLOUD
#define VECFIELD_NAME				"cloud"
#define VECFIELD_PATH				"E:/data/flow/Cloud_49.vec"
#define VECHEADER_PATH				"E:/data/flow/Cloud.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined SQUARE_CYLINDER
#define VECFIELD_NAME				"cylinder"
#define VECFIELD_PATH				"E:/data/flow/cylinder.vec"
#define VECHEADER_PATH				"E:/data/flow/cylinder.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined ABC_FLOW
#define VECFIELD_NAME				"abc"
#define VECFIELD_PATH				"E:/data/flow/abc.vec"
#define VECHEADER_PATH				"E:/data/flow/abc.hdr"
#define	SEG_LEN						1.0f
#define TUBE_RADIUS					0.2f
#elif defined ATMOSPHERICAL
#define VECFIELD_NAME				"atmospherical"
#define VECFIELD_PATH				"E:/data/atmospherical/VEC/053.dat"
#define VECHEADER_PATH				"E:/data/atmospherical/atmospherical.hdr"
#define	SEG_LEN						1.0f
#define TUBE_RADIUS					0.2f
#elif defined OSU_DNS
#define VECFIELD_NAME				"osu_dns"
#define VECFIELD_PATH				"E:/data/OSU/DNS_DATA_CHOPPED/V/V.000006"
#define VECHEADER_PATH				"E:/data/OSU/DNS_DATA_CHOPPED/DNS.hdr"
#define	SEG_LEN						2.0f
#define TUBE_RADIUS					0.2f
#elif defined LES_JET
#define VECFIELD_NAME				"les-jet"
#define VECFIELD_PATH				"E:/data/OSU/LES-JET-NO-HEADER/V/V.000049"
#define VECHEADER_PATH				"E:/data/OSU/LES-JET-NO-HEADER/les-jet.hdr"
#define	SEG_LEN						1.0f
#define TUBE_RADIUS					0.2f
#elif defined GL3D
#define VECFIELD_NAME				"GL3D"
#define VECFIELD_PATH				"E:/data/flow/GL3D.vec"
#define VECHEADER_PATH				"E:/data/flow/GL3D.hdr"
#define	SEG_LEN						1.0f
#define TUBE_RADIUS					0.2f
#elif defined GL3D2
#define VECFIELD_NAME				"GL3D2"
#define VECFIELD_PATH				"E:/data/flow/GL3D2.vec"
#define VECHEADER_PATH				"E:/data/flow/GL3D.hdr"
#define	SEG_LEN						1.0f
#define TUBE_RADIUS					0.2f
#endif


#define SAVE_IMAGE_PATH         "./SavedImages/"

#define MAX_USER_PATH_POINTS    256

//TUBE
#define TUBE_NUM_FACE			8

//BINS
#define NUM_MAG_BIN				4
#define NUM_SPHERE_BIN			50
#define NUM_3D_BIN				200
#define ENTROPY_WIN_SIZE		4

//JITTER
#define ACSIZE	8

#endif//MY_DEFINITION