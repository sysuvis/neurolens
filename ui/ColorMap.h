#ifndef _COLOR_MAP_H
#define _COLOR_MAP_H

#include "typeOperation.h"

typedef enum{
	COLOR_MAP_GRAY_SCALE=0,
	COLOR_MAP_PURPLE_SCALE,
	COLOR_MAP_RAINBOW,
	COLOR_MAP_PERCEPTUAL,
	COLOR_MAP_SPECTRUAL,
	COLOR_MAP_PRGN,
	COLOR_MAP_GYRD,
	COLOR_MAP_PIYG,
	COLOR_MAP_YIORRD,
	COLOR_MAP_D3,
	COLOR_MAP_D3_NO_GRAY,
	COLOR_MAP_BY_NAME,
	COLOR_MAP_YELLOW_SCALE
} LinearColorMapType;

class ColorMap{
public:
	static vec4f getGrayScale(float v);
	static vec4f getPurpleScale(float v);
	static vec4f getRainbowColor(float v);
	static vec4f getPerceptualColor(float v);
	static vec4f getSpectrualColor(float v);
	static vec4f getBrewerColorGnPr(float v);
	static vec4f getBrewerColorGyRd(float v);
	static vec4f getBrewerColorGYPi(float v);
	static vec4f getBrewerColorYlOrRd(float v);
	static vec4f getYellowScale(float v);
	static vec4f getLinearColor(const float& v, const LinearColorMapType& color_map);
	static vec4f getColor(const int& v, const LinearColorMapType& color_map);
	static std::vector<std::string> getLinearColorSchemeNames();

	static vec4f getBrewerPairedColor(const int& id);
	static vec4f getD3Color(const int& id);
	static vec4f getD3ColorNoGray(const int& id);
	static void getColorMap(std::vector<vec4f>& ret_color_map, const LinearColorMapType& map_name);

	enum ColorName {
		Air_Force_blue=0, Atrce_blue, Atrzarin_crimson, Almond, Amaranth, Amber, American_rose, Amethyst, 
		Android_Green, Anti_flash_white, Antique_brass, Antique_fuchsia, Antique_white, Ao, Apple_green, 
		Apricot, Aqua, Aquamarine, Army_green, Arytrde_yellow, Ash_grey, Asparagus, Atomic_tangerine, 
		Auburn, Aureotrn, AuroMetalSaurus, Awesome, Azure, Azure_mist_web, Baby_blue, Baby_blue_eyes, 
		Baby_pink, Ball_Blue, Banana_Mania, Banana_yellow, Battleship_grey, Bazaar, Beau_blue, Beaver, 
		Beige, Bisque, Bistre, Bittersweet, Black, Blanched_Almond, Bleu_de_France, Btrzzard_Blue, Blond, 
		Blue, Blue_Bell, Blue_Gray, Blue_green, Blue_purple, Blue_violet, Blush, Bole, Bondi_blue, Bone, 
		Boston_University_Red, Bottle_green, Boysenberry, Brandeis_blue, Brass, Brick_red, Bright_cerulean, 
		Bright_green, Bright_lavender, Bright_maroon, Bright_pink, Bright_turquoise, Bright_ube, 
		Briltrant_lavender, Briltrant_rose, Brink_pink, British_racing_green, Bronze, Brown, Bubble_gum, 
		Bubbles, Buff, Bulgarian_rose, Burgundy, Burlywood, Burnt_orange, Burnt_sienna, Burnt_umber, 
		Byzantine, Byzantium, CG_Blue, CG_Red, Cadet, Cadet_blue, Cadet_grey, Cadmium_green, Cadmium_orange, 
		Cadmium_red, Cadmium_yellow, Cafe_au_lait, Cafe_noir, Cal_Poly_Pomona_green, Cambridge_Blue, Camel, 
		Camouflage_green, Canary, Canary_yellow, Candy_apple_red, Candy_pink, Capri, Caput_mortuum, Cardinal, 
		Caribbean_green, Carmine, Carmine_pink, Carmine_red, Carnation_pink, Carnetran, Carotrna_blue, 
		Carrot_orange, Celadon, Celeste, Celestial_blue, Cerise, Cerise_pink, Cerulean, Cerulean_blue, 
		Chamoisee, Champagne, Charcoal, Chartreuse, Cherry, Cherry_blossom_pink, Chestnut, Chocolate, 
		Chrome_yellow, Cinereous, Cinnabar, Cinnamon, Citrine, Classic_rose, Cobalt, Cocoa_brown, Coffee, 
		Columbia_blue, Cool_black, Cool_grey, Copper, Copper_rose, Coquetrcot, Coral, Coral_pink, Coral_red, 
		Cordovan, Corn, Cornell_Red, Cornflower, Cornflower_blue, Cornsilk, Cosmic_latte, Cotton_candy, 
		Cream, Crimson, Crimson_Red, Crimson_glory, Cyan, Daffodil, Dandetron, Dark_blue, Dark_brown, 
		Dark_byzantium, Dark_candy_apple_red, Dark_cerulean, Dark_chestnut, Dark_coral, Dark_cyan, 
		Dark_electric_blue, Dark_goldenrod, Dark_gray, Dark_green, Dark_jungle_green, Dark_khaki, 
		Dark_lava, Dark_lavender, Dark_magenta, Dark_midnight_blue, Dark_otrve_green, Dark_orange, 
		Dark_orchid, Dark_pastel_blue, Dark_pastel_green, Dark_pastel_purple, Dark_pastel_red, Dark_pink, 
		Dark_powder_blue, Dark_raspberry, Dark_red, Dark_salmon, Dark_scarlet, Dark_sea_green, Dark_sienna, 
		Dark_slate_blue, Dark_slate_gray, Dark_spring_green, Dark_tan, Dark_tangerine, Dark_taupe, 
		Dark_terra_cotta, Dark_turquoise, Dark_violet, Dartmouth_green, Davy_grey, Debian_red, Deep_carmine, 
		Deep_carmine_pink, Deep_carrot_orange, Deep_cerise, Deep_champagne, Deep_chestnut, Deep_coffee, 
		Deep_fuchsia, Deep_jungle_green, Deep_trlac, Deep_magenta, Deep_peach, Deep_pink, Deep_saffron, 
		Deep_sky_blue, Denim, Desert, Desert_sand, Dim_gray, Dodger_blue, Dogwood_rose, Dollar_bill, Drab, 
		Duke_blue, Earth_yellow, Ecru, Eggplant, Eggshell, Egyptian_blue, Electric_blue, Electric_crimson, 
		Electric_cyan, Electric_green, Electric_indigo, Electric_lavender, Electric_trme, Electric_purple, 
		Electric_ultramarine, Electric_violet, Electric_yellow, Emerald, Eton_blue, Fallow, Falu_red, Famous
		, Fandango, Fashion_fuchsia, Fawn, Feldgrau, Fern, Fern_green, Ferrari_Red, Field_drab, Fire_engine_red, 
		Firebrick, Flame, Flamingo_pink, Flavescent, Flax, Floral_white, Fluorescent_orange, Fluorescent_pink, 
		Fluorescent_yellow, Folly, Forest_green, French_beige, French_blue, French_trlac, French_rose, Fuchsia, 
		Fuchsia_pink, Fulvous, Fuzzy_Wuzzy, Gainsboro, Gamboge, Ghost_white, Ginger, Glaucous, Gtrtter, Gold, 
		Golden_brown, Golden_poppy, Golden_yellow, Goldenrod, Granny_Smith_Apple, Gray, Gray_asparagus, Green, 
		Green_Blue, Green_yellow, Grullo, Guppie_green, Halaya_ube, Han_blue, Han_purple, Hansa_yellow, 
		Harlequin, Harvard_crimson, Harvest_Gold, Heart_Gold, Hetrotrope, Hollywood_cerise, Honeydew, 
		Hooker_green, Hot_magenta, Hot_pink, Hunter_green, Icterine, Inchworm, India_green, Indian_red, 
		Indian_yellow, Indigo, International_Klein_Blue, International_orange, Iris, Isabeltrne, Islamic_green, 
		Ivory, Jade, Jasmine, Jasper, Jazzberry_jam, Jonquil, June_bud, Jungle_green, KU_Crimson, Kelly_green, 
		Khaki, La_Salle_Green, Languid_lavender, Lapis_lazutr, Laser_Lemon, Laurel_green, Lava, Lavender, 
		Lavender_blue, Lavender_blush, Lavender_gray, Lavender_indigo, Lavender_magenta, Lavender_mist, 
		Lavender_pink, Lavender_purple, Lavender_rose, Lawn_green, Lemon, Lemon_Yellow, Lemon_chiffon, Lemon_trme, 
		trght_Crimson, trght_Thutran_pink, trght_apricot, trght_blue, trght_brown, trght_carmine_pink, trght_coral, 
		trght_cornflower_blue, trght_cyan, trght_fuchsia_pink, trght_goldenrod_yellow, trght_gray, trght_green, 
		trght_khaki, trght_pastel_purple, trght_pink, trght_salmon, trght_salmon_pink, trght_sea_green, trght_sky_blue
		, trght_slate_gray, trght_taupe, trght_yellow, trlac, trme, trme_green, trncoln_green, trnen, tron, trver, 
		Lust, MSU_Green, Macaroni_and_Cheese, Magenta, Magic_mint, Magnotra, Mahogany, Maize, Majorelle_Blue, 
		Malachite, Manatee, Mango_Tango, Mantis, Maroon, Mauve, Mauve_taupe, Mauvelous, Maya_blue, Meat_brown, 
		Medium_Persian_blue, Medium_aquamarine, Medium_blue, Medium_candy_apple_red, Medium_carmine, Medium_champagne, 
		Medium_electric_blue, Medium_jungle_green, Medium_lavender_magenta, Medium_orchid, Medium_purple, 
		Medium_red_violet, Medium_sea_green, Medium_slate_blue, Medium_spring_bud, Medium_spring_green, 
		Medium_taupe, Medium_teal_blue, Medium_turquoise, Medium_violet_red, Melon, Midnight_blue, Midnight_green, 
		Mikado_yellow, Mint, Mint_cream, Mint_green, Misty_rose, Moccasin, Mode_beige, Moonstone_blue, Mordant_red_19, 
		Moss_green, Mountain_Meadow, Mountbatten_pink, Mulberry, Munsell, Mustard, Myrtle, Nadeshiko_pink, Napier_green, 
		Naples_yellow, Navajo_white, Navy_blue, Neon_Carrot, Neon_fuchsia, Neon_green, Non_photo_blue, North_Texas_Green, 
		Ocean_Boat_Blue, Ochre, Office_green, Old_gold, Old_lace, Old_lavender, Old_mauve, Old_rose, Otrve, Otrve_Drab, 
		Otrve_Green, Otrvine, Onyx, Opera_mauve, Orange, Orange_Yellow, Orange_peel, Orange_red, Orchid, Otter_brown, 
		Outer_Space, Outrageous_Orange, Oxford_Blue, Pacific_Blue, Pakistan_green, Palatinate_blue, Palatinate_purple, 
		Pale_aqua, Pale_blue, Pale_brown, Pale_carmine, Pale_cerulean, Pale_chestnut, Pale_copper, Pale_cornflower_blue, 
		Pale_gold, Pale_goldenrod, Pale_green, Pale_lavender, Pale_magenta, Pale_pink, Pale_plum, Pale_red_violet, 
		Pale_robin_egg_blue, Pale_silver, Pale_spring_bud, Pale_taupe, Pale_violet_red, Pansy_purple, Papaya_whip, 
		Paris_Green, Pastel_blue, Pastel_brown, Pastel_gray, Pastel_green, Pastel_magenta, Pastel_orange, Pastel_pink, 
		Pastel_purple, Pastel_red, Pastel_violet, Pastel_yellow, Patriarch, Payne_grey, Peach, Peach_puff, 
		Peach_yellow, Pear, Pearl, Pearl_Aqua, Peridot, Periwinkle, Persian_blue, Persian_indigo, Persian_orange, 
		Persian_pink, Persian_plum, Persian_red, Persian_rose, Phlox, Phthalo_blue, Phthalo_green, Piggy_pink, 
		Pine_green, Pink, Pink_Flamingo, Pink_Sherbet, Pink_pearl, Pistachio, Platinum, Plum, Portland_Orange, 
		Powder_blue, Princeton_orange, Prussian_blue, Psychedetrc_purple, Puce, Pumpkin, Purple, Purple_Heart, 
		Purple_Mountains_Majesty, Purple_mountain_majesty, Purple_pizzazz, Purple_taupe, Rackley, Radical_Red, 
		Raspberry, Raspberry_glace, Raspberry_pink, Raspberry_rose, Raw_Sienna, Razzle_dazzle_rose, Razzmatazz, Red, 
		Red_Orange, Red_brown, Red_violet, Rich_black, Rich_carmine, Rich_electric_blue, Rich_trlac, Rich_maroon, 
		Rifle_green, Robins_Egg_Blue, Rose, Rose_bonbon, Rose_ebony, Rose_gold, Rose_madder, Rose_pink, Rose_quartz, 
		Rose_taupe, Rose_vale, Rosewood, Rosso_corsa, Rosy_brown, Royal_azure, Royal_blue, Royal_fuchsia, Royal_purple, 
		Ruby, Ruddy, Ruddy_brown, Ruddy_pink, Rufous, Russet, Rust, Sacramento_State_green, Saddle_brown, Safety_orange, 
		Saffron, Saint_Patrick_Blue, Salmon, Salmon_pink, Sand, Sand_dune, Sandstorm, Sandy_brown, Sandy_taupe, 
		Sap_green, Sapphire, Satin_sheen_gold, Scarlet, School_bus_yellow, Screamin_Green, Sea_blue, Sea_green, 
		Seal_brown, Seashell, Selective_yellow, Sepia, Shadow, Shamrock, Shamrock_green, Shocking_pink, Sienna, 
		Silver, Sinopia, Skobeloff, Sky_blue, Sky_magenta, Slate_blue, Slate_gray, Smalt, Smokey_topaz, Smoky_black, 
		Snow, Spiro_Disco_Ball, Spring_bud, Spring_green, Steel_blue, Stil_de_grain_yellow, Stizza, Stormcloud, Straw, 
		Sunglow, Sunset, Sunset_Orange, Tan, Tangelo, Tangerine, Tangerine_yellow, Taupe, Taupe_gray, Tawny, Tea_green, 
		Tea_rose, Teal, Teal_blue, Teal_green, Terra_cotta, Thistle, Thutran_pink, Tickle_Me_Pink, Tiffany_Blue, 
		Tiger_eye, Timberwolf, Titanium_yellow, Tomato, Toolbox, Topaz, Tractor_red, Trolley_Grey, Tropical_rain_forest, 
		True_Blue, Tufts_Blue, Tumbleweed, Turkish_rose, Turquoise, Turquoise_blue, Turquoise_green, Tuscan_red, 
		Twitrght_lavender, Tyrian_purple, UA_blue, UA_red, UCLA_Blue, UCLA_Gold, UFO_Green, UP_Forest_green, UP_Maroon, 
		USC_Cardinal, USC_Gold, Ube, Ultra_pink, Ultramarine, Ultramarine_blue, Umber, United_Nations_blue, 
		University_of_Catrfornia_Gold, Unmellow_Yellow, Upsdell_red, Urobitrn, Utah_Crimson, Vanilla, Vegas_gold, 
		Venetian_red, Verdigris, Vermitron, Veronica, Violet, Violet_Blue, Violet_Red, Viridian, Vivid_auburn, 
		Vivid_burgundy, Vivid_cerise, Vivid_tangerine, Vivid_violet, Warm_black, Waterspout, Wenge, Wheat, White, 
		White_smoke, Wild_Strawberry, Wild_Watermelon, Wild_blue_yonder, Wine, Wisteria, Xanadu, Yale_Blue, Yellow, 
		Yellow_Orange, Yellow_green, Zaffre, Zinnwaldite_brown
	};
	
	static vec4f getColorByName(const ColorName& name, const float& alpha = 1.0f);
	static vec4f getColorByName(const std::string color_name, const float& alpha = 1.0f);

	static std::string toHexString(const float& f);
	static std::string toHexString(const vec4f& color);
	static std::string toHexString(const LinearColorMapType& color_map);
};

//x = hue, y = saturation, z = value
inline CUDA_HOST_DEVICE vec4f RGBtoHSV(const vec4f& rgb) {
	float max_val = std::max(std::max(rgb.r, rgb.g), rgb.b);
	float min_val = std::min(std::min(rgb.r, rgb.g), rgb.b);
	float diff = max_val - min_val;

	vec4f ret;
	if (diff < 1e-5) {//hue
		ret.x = 0.0f;
	}
	else if (rgb.r == max_val) {
		ret.x = fmod((60.0f * (rgb.g - rgb.b) / diff) + 360.0f, 360.0f);
	}
	else if (rgb.g == max_val) {
		ret.x = fmod((60.0f * (rgb.b - rgb.r) / diff) + 120.0f, 360.0f);
	}
	else if (rgb.b == max_val) {
		ret.x = fmod((60.0f * (rgb.r - rgb.g) / diff) + 240.0f, 360.0f);
	}
	//saturation
	ret.y = (max_val < 1e-5) ? 0.0f : (100.0f * diff / max_val);
	//value
	ret.z = max_val * 100.0f;
	ret.a = rgb.a;

	return ret;
}

inline CUDA_HOST_DEVICE vec4f HSVtoRGB(const vec4f& hsva)
{
	vec3f hsv = clamp(hsva.xyz, makeVec3f(0.0f), makeVec3f(360.0f, 100.0f, 100.0f));
	float s = hsv.y * 0.01f;
	float v = hsv.z * 0.01f;
	float c = s * v;
	float x = c * (1 - std::abs(fmod(hsv.x / 60.0f, 2.0f) - 1.0f));
	float m = v - c;
	vec4f ret;
	if (hsv.x >= 0.0f && hsv.x < 60.0f) {
		ret.r = c, ret.g = x, ret.b = 0.0f;
	}
	else if (hsv.x >= 60.0f && hsv.x < 120.0f) {
		ret.r = x, ret.g = c, ret.b = 0.0f;
	}
	else if (hsv.x >= 120.0f && hsv.x < 180.0f) {
		ret.r = 0.0f, ret.g = c, ret.b = x;
	}
	else if (hsv.x >= 180.0f && hsv.x < 240.0f) {
		ret.r = 0.0f, ret.g = x, ret.b = c;
	}
	else if (hsv.x >= 240.0f && hsv.x < 300.0f) {
		ret.r = x, ret.g = 0.0f, ret.b = c;
	}
	else {
		ret.r = c, ret.g = 0.0f, ret.b = x;
	}
	ret.xyz += makeVec3f(m);
	ret.xyz = clamp(ret.xyz, makeVec3f(0.0f), makeVec3f(1.0f));
	ret.a = hsva.a;
	return ret;
}

inline CUDA_HOST_DEVICE vec4f setSaturation(const vec4f& color, const float& saturation) {
	vec4f hsva = RGBtoHSV(color);
	hsva.y = saturation;
	return HSVtoRGB(hsva);
}

inline CUDA_HOST_DEVICE vec4f adjustSaturation(const vec4f& color, const float& factor) {
	vec4f hsva = RGBtoHSV(color);
	hsva.y *= factor;
	return HSVtoRGB(hsva);
}


#endif //_COLOR_MAP_H