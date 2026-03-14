"""
Music Discovery Engine — Data Indexer
Fetches a music dataset, generates embeddings, and upserts to Endee Vector DB.
"""
import os
import time
import json
import csv
import io
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── Configuration ──────────────────────────────────────────────────────────────
ENDEE_HOST = os.environ.get("ENDEE_HOST", "http://localhost:8080")
INDEX_NAME = "music"
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension
BATCH_SIZE = 100

# ── Curated music dataset ─────────────────────────────────────────────────────
# We bundle a rich sample dataset inline so the project works out of the box
# without needing an external download. Each track has: title, artist, album,
# genre, mood/vibe tags, year, and a short description/lyrics snippet.

MUSIC_DATA = [
    # ── Chill / Lo-fi ──────────────────────────────────────────────────────────
    {"id": "t001", "title": "Midnight Rain", "artist": "Lofi Dreamer", "album": "Dawn Sessions", "genre": "Lo-fi", "mood": "Chill, Rainy, Mellow", "year": "2023", "description": "Soft piano loops over gentle rain sounds, perfect for late-night studying or winding down after a long day. Warm vinyl crackle and distant thunder."},
    {"id": "t002", "title": "Velvet Clouds", "artist": "Sleepy Beats", "album": "Driftwood", "genre": "Lo-fi", "mood": "Dreamy, Peaceful, Floating", "year": "2022", "description": "Ethereal pads layered with soft guitar picking, creating a dreamlike atmosphere that feels like floating through clouds at sunset."},
    {"id": "t003", "title": "Coffee & Pages", "artist": "Jazz Hop Cafe", "album": "Morning Brew", "genre": "Lo-fi Jazz", "mood": "Cozy, Warm, Relaxed", "year": "2023", "description": "Smooth jazz saxophone over boom-bap beats, like reading a book in a warm cafe on a cold morning with steaming coffee."},
    {"id": "t004", "title": "Sunset Boulevard", "artist": "Chillhop Essentials", "album": "Golden Hour", "genre": "Chillhop", "mood": "Warm, Golden, Nostalgic", "year": "2021", "description": "Mellow guitar riffs and soft drums capturing the warmth of a California sunset drive with the windows down."},
    {"id": "t005", "title": "Rain on Glass", "artist": "Ambient Waves", "album": "Watercolors", "genre": "Ambient", "mood": "Calm, Meditative, Rainy", "year": "2022", "description": "Delicate ambient textures with actual rain recordings, designed for deep focus and meditation during stormy weather."},

    # ── Late Night / Drive ─────────────────────────────────────────────────────
    {"id": "t006", "title": "Neon Highways", "artist": "Synthwave Runner", "album": "After Dark", "genre": "Synthwave", "mood": "Energetic, Night Drive, Retro", "year": "2023", "description": "Pulsing synthesizers and retro drum machines that evoke speeding through a neon-lit city at 2 AM with the top down."},
    {"id": "t007", "title": "Starlight Express", "artist": "Midnight Cruiser", "album": "Cosmic Lanes", "genre": "Synthwave", "mood": "Dreamy, Night, Cosmic", "year": "2022", "description": "Lush analog synths and arpeggios gliding through a starlit highway, blending 80s nostalgia with modern production."},
    {"id": "t008", "title": "City Lights Below", "artist": "Night Owl", "album": "Urban Glow", "genre": "Electronic", "mood": "Atmospheric, Urban, Night", "year": "2023", "description": "Deep bass and shimmering pads painting the view of a city skyline from a rooftop at midnight, lights twinkling below."},
    {"id": "t009", "title": "Empty Freeway", "artist": "Drive Mode", "album": "Highway One", "genre": "Electronic Chill", "mood": "Solo, Open Road, Reflective", "year": "2021", "description": "Minimal beats and wide stereo soundscapes capturing the solitude and freedom of driving an empty highway at dawn."},
    {"id": "t010", "title": "Twilight Zone", "artist": "Retro Pulse", "album": "Neon Dreams", "genre": "Retrowave", "mood": "Mysterious, Night, Moody", "year": "2022", "description": "Dark synth layers and haunting melodies creating a mysterious atmosphere, like wandering through a foggy neon-lit alley."},

    # ── Workout / Energy ───────────────────────────────────────────────────────
    {"id": "t011", "title": "Beast Mode", "artist": "Power Surge", "album": "Iron Will", "genre": "EDM", "mood": "Intense, Pump Up, Aggressive", "year": "2023", "description": "Hard-hitting bass drops and aggressive synths designed to push you through the toughest workout sets. Pure adrenaline fuel."},
    {"id": "t012", "title": "Thunder Strike", "artist": "Bass Cannon", "album": "Voltage", "genre": "Dubstep", "mood": "Heavy, Explosive, Raw", "year": "2022", "description": "Massive dubstep drops and distorted bass that hits like lightning. Made for heavy lifting and intense cardio sessions."},
    {"id": "t013", "title": "Run the World", "artist": "Tempo Rising", "album": "Finish Line", "genre": "Electronic Pop", "mood": "Motivational, Running, Uplifting", "year": "2023", "description": "Uplifting synth melodies over driving four-on-the-floor beats, perfect tempo for running and high-energy cardio."},
    {"id": "t014", "title": "Adrenaline Rush", "artist": "Rave Factory", "album": "Peak Hours", "genre": "Hardstyle", "mood": "Extreme, Rave, Unstoppable", "year": "2021", "description": "Hardstyle kicks and euphoric leads that transport you to a massive festival main stage. Pure energy overload."},
    {"id": "t015", "title": "No Limits", "artist": "Gym Anthem", "album": "Unbreakable", "genre": "Hip-Hop/EDM", "mood": "Confident, Grind, Hustle", "year": "2023", "description": "Trap-influenced beats with motivational vocal chops about pushing through barriers and reaching your peak potential."},

    # ── Romantic / Emotional ───────────────────────────────────────────────────
    {"id": "t016", "title": "First Touch", "artist": "Heartstrings", "album": "Velvet Nights", "genre": "R&B", "mood": "Romantic, Intimate, Tender", "year": "2022", "description": "Silky smooth R&B vocals over warm keys and soft percussion, capturing the electricity of a first romantic encounter."},
    {"id": "t017", "title": "Moonlit Dance", "artist": "Evening Rose", "album": "Whispered Love", "genre": "Jazz", "mood": "Romantic, Elegant, Slow Dance", "year": "2023", "description": "Classic jazz arrangement with brushed drums and warm trumpet, like a slow dance under moonlight at an intimate venue."},
    {"id": "t018", "title": "Letters Never Sent", "artist": "Acoustic Soul", "album": "Paper Hearts", "genre": "Indie Folk", "mood": "Melancholy, Bittersweet, Nostalgic", "year": "2022", "description": "Fingerpicked acoustic guitar and vulnerable vocals telling a story of unspoken love and words left unsaid."},
    {"id": "t019", "title": "Cherry Blossoms", "artist": "Spring Melody", "album": "Garden of Love", "genre": "Pop", "mood": "Sweet, Joyful, Fresh Love", "year": "2023", "description": "Bright pop production with cheerful synths and upbeat rhythm capturing the giddiness and wonder of falling in love."},
    {"id": "t020", "title": "Fading Polaroids", "artist": "Indie Hearts", "album": "Yesterday's Colors", "genre": "Indie Pop", "mood": "Nostalgic, Wistful, Dreamy", "year": "2021", "description": "Hazy reverb-drenched guitars and breathy vocals reminiscing about summer love that faded like old photographs."},

    # ── Focus / Study ──────────────────────────────────────────────────────────
    {"id": "t021", "title": "Deep Focus", "artist": "Concentration FM", "album": "Flow State", "genre": "Ambient Electronic", "mood": "Focused, Productive, Minimal", "year": "2023", "description": "Precisely crafted ambient textures with binaural undertones optimized for deep work and sustained concentration."},
    {"id": "t022", "title": "Library Whispers", "artist": "Study Beats", "album": "Academic Hours", "genre": "Lo-fi", "mood": "Quiet, Studious, Gentle", "year": "2022", "description": "Ultra-soft lo-fi beats with page-turning sounds and distant library ambiance, perfect for all-night study sessions."},
    {"id": "t023", "title": "Algorithm", "artist": "Code Flow", "album": "Debug Mode", "genre": "Electronic", "mood": "Technical, Focused, Mechanical", "year": "2023", "description": "Clean, precise electronic patterns with evolving sequences that mirror the rhythm of coding and problem-solving."},
    {"id": "t024", "title": "Ocean Mind", "artist": "Nature Loops", "album": "Blue Horizons", "genre": "Ambient", "mood": "Zen, Oceanic, Clarity", "year": "2022", "description": "Gentle ocean waves blended with subtle synth pads creating an expansive mental space for creative thinking."},
    {"id": "t025", "title": "Morning Clarity", "artist": "Dawn Tones", "album": "First Light", "genre": "Classical Ambient", "mood": "Fresh, Clear, Awakening", "year": "2021", "description": "Soft piano and string arrangements paired with morning birdsong, designed for productive early morning routines."},

    # ── Party / Celebration ────────────────────────────────────────────────────
    {"id": "t026", "title": "Weekend Fever", "artist": "Club Kings", "album": "Main Floor", "genre": "House", "mood": "Party, Euphoric, Dancing", "year": "2023", "description": "Infectious house groove with funky basslines and vocal hooks that command the dance floor on a Saturday night."},
    {"id": "t027", "title": "Champagne Showers", "artist": "VIP Lounge", "album": "Luxe Life", "genre": "Deep House", "mood": "Luxurious, Celebration, Smooth", "year": "2022", "description": "Sleek deep house with jazzy chord progressions and refined production, like popping bottles at a rooftop party."},
    {"id": "t028", "title": "Disco Revival", "artist": "Groove Machine", "album": "Studio 54", "genre": "Nu-Disco", "mood": "Funky, Retro Party, Groovy", "year": "2023", "description": "Funk guitar riffs and disco strings revived with modern production, bringing back the golden era of dance music."},
    {"id": "t029", "title": "Bass Drop City", "artist": "Festival Sound", "album": "Mainstage", "genre": "EDM", "mood": "Festival, Epic, Crowd", "year": "2021", "description": "Massive festival anthem with euphoric buildups and earth-shaking drops that unite thousands of hands in the air."},
    {"id": "t030", "title": "Summer Anthem", "artist": "Beach Vibes", "album": "Tropic Thunder", "genre": "Tropical House", "mood": "Summer, Tropical, Carefree", "year": "2023", "description": "Tropical house with steel drums and marimba layered over four-on-the-floor beats, instant summer vacation mood."},

    # ── Sad / Melancholic ──────────────────────────────────────────────────────
    {"id": "t031", "title": "Empty Room", "artist": "Solitary Piano", "album": "Silence Speaks", "genre": "Classical", "mood": "Sad, Lonely, Contemplative", "year": "2022", "description": "A lone piano playing in an empty concert hall, each note echoing with the weight of solitude and unspoken grief."},
    {"id": "t032", "title": "After the Rain", "artist": "Grey Skies", "album": "November", "genre": "Post-Rock", "mood": "Melancholy, Atmospheric, Healing", "year": "2023", "description": "Building layers of guitars and delayed textures that swell from quiet sadness to cathartic emotional release."},
    {"id": "t033", "title": "3 AM Thoughts", "artist": "Insomnia Club", "album": "Sleepless", "genre": "R&B", "mood": "Vulnerable, Late Night, Raw", "year": "2022", "description": "Raw, confessional R&B about lying awake at 3 AM replaying mistakes and wondering what could have been different."},
    {"id": "t034", "title": "Autumn Leaves Fall", "artist": "Acoustic Tears", "album": "Seasons End", "genre": "Folk", "mood": "Bittersweet, Autumn, Reflective", "year": "2021", "description": "Gentle folk fingerpicking with cello accompaniment painting scenes of falling leaves and the beauty of letting go."},
    {"id": "t035", "title": "Ghost of You", "artist": "Memory Lane", "album": "Echoes", "genre": "Indie", "mood": "Haunting, Loss, Yearning", "year": "2023", "description": "Reverb-soaked guitars and distant vocals that feel like chasing a memory that keeps slipping further away."},

    # ── Nature / Adventure ─────────────────────────────────────────────────────
    {"id": "t036", "title": "Mountain Summit", "artist": "Trail Blazer", "album": "Altitude", "genre": "Epic Orchestral", "mood": "Triumphant, Adventure, Majestic", "year": "2023", "description": "Sweeping orchestral arrangement with soaring horns and thundering timpani, capturing the triumph of reaching a mountain peak."},
    {"id": "t037", "title": "Forest Bathing", "artist": "Woodland Spirit", "album": "Green Cathedral", "genre": "Ambient Nature", "mood": "Peaceful, Forest, Grounded", "year": "2022", "description": "Layered nature recordings with subtle musical elements creating an immersive walk through an ancient forest at dawn."},
    {"id": "t038", "title": "Ocean Drift", "artist": "Seafarer", "album": "Tidal Patterns", "genre": "Ambient Electronic", "mood": "Vast, Oceanic, Adventurous", "year": "2023", "description": "Expansive electronic soundscapes inspired by open ocean sailing, with waves of synth and rhythmic pulses like tides."},
    {"id": "t039", "title": "Desert Stars", "artist": "Nomad Sound", "album": "Sahara Nights", "genre": "World Electronic", "mood": "Mystical, Desert, Exotic", "year": "2021", "description": "Middle Eastern scales blended with electronic production under a vast starlit desert sky, evoking ancient caravan stories."},
    {"id": "t040", "title": "Northern Lights", "artist": "Arctic Dreams", "album": "Aurora", "genre": "Ambient", "mood": "Magical, Cold, Ethereal", "year": "2022", "description": "Shimmering crystalline synths and icy reverbs painting the breathtaking colors of the aurora borealis dancing across the Arctic sky."},

    # ── Hip-Hop / Urban ────────────────────────────────────────────────────────
    {"id": "t041", "title": "Street Poetry", "artist": "Urban Wordsmith", "album": "Concrete Jungle", "genre": "Hip-Hop", "mood": "Raw, Street, Real", "year": "2023", "description": "Gritty boom-bap beats with sharp lyrical storytelling about life in the city, hustle, and keeping it authentic."},
    {"id": "t042", "title": "Cloud Nine", "artist": "Smooth Operator", "album": "Elevated", "genre": "Cloud Rap", "mood": "Spacey, Floating, Hazy", "year": "2022", "description": "Dreamy trap beats with reverb-heavy vocals floating through atmospheric synths like drifting through clouds."},
    {"id": "t043", "title": "Gold Chains", "artist": "Flex Mode", "album": "Status", "genre": "Trap", "mood": "Confident, Luxurious, Bold", "year": "2023", "description": "Heavy 808s and hi-hat rolls with confident delivery about success, ambition, and the fruits of hard work."},
    {"id": "t044", "title": "Old School Joint", "artist": "Boom Bap Revival", "album": "90s Forever", "genre": "Hip-Hop", "mood": "Nostalgic, Classic, Groovy", "year": "2021", "description": "Classic 90s-inspired boom-bap with vinyl scratching, jazzy samples, and clever wordplay paying homage to golden era hip-hop."},
    {"id": "t045", "title": "Midnight Hustle", "artist": "Grind Time", "album": "No Sleep", "genre": "Trap", "mood": "Hustle, Grind, Dark", "year": "2022", "description": "Dark, moody trap production with minimal lyrics about late-night grinding, ambition, and sacrifice for success."},

    # ── Classical / Cinematic ──────────────────────────────────────────────────
    {"id": "t046", "title": "The Grand Finale", "artist": "Symphony Orchestra", "album": "Opus Magnum", "genre": "Classical", "mood": "Epic, Grand, Climactic", "year": "2023", "description": "Full orchestral symphony building to a massive climax with every instrument joining in a triumphant, goosebump-inducing finale."},
    {"id": "t047", "title": "Requiem for Dreams", "artist": "Film Score Studio", "album": "Cinematic Journeys", "genre": "Film Score", "mood": "Dramatic, Cinematic, Emotional", "year": "2022", "description": "Hauntingly beautiful string arrangement with choir vocals growing from a whisper to a powerful emotional crescendo."},
    {"id": "t048", "title": "Gentle Waltz", "artist": "Piano Stories", "album": "Storybook", "genre": "Classical Piano", "mood": "Elegant, Whimsical, Charming", "year": "2023", "description": "Delicate piano waltz with a fairytale quality, like music playing in an old European ballroom filled with candlelight."},
    {"id": "t049", "title": "Battle Cry", "artist": "War Drums", "album": "Legends Rise", "genre": "Epic Trailer", "mood": "Powerful, Heroic, Battle", "year": "2021", "description": "Thundering percussion and brass building an intense battle theme with choir chants, designed for epic movie moments."},
    {"id": "t050", "title": "Lullaby in D Minor", "artist": "String Quartet", "album": "Goodnight Moon", "genre": "Chamber Music", "mood": "Tender, Sleepy, Soothing", "year": "2022", "description": "Intimate string quartet playing a gentle lullaby that wraps around you like a warm blanket on a quiet evening."},

    # ── Rock / Alternative ─────────────────────────────────────────────────────
    {"id": "t051", "title": "Rebel Heart", "artist": "Garage Riot", "album": "Loud & Clear", "genre": "Rock", "mood": "Rebellious, Raw, Energetic", "year": "2023", "description": "Distorted guitars and pounding drums channeling pure punk energy and the spirit of rebellion against the mundane."},
    {"id": "t052", "title": "Echoes of Tomorrow", "artist": "Alt Horizon", "album": "Future Past", "genre": "Alternative", "mood": "Introspective, Hopeful, Modern", "year": "2022", "description": "Layered alternative rock with shimmering guitars and introspective lyrics about finding hope in uncertain times."},
    {"id": "t053", "title": "Grunge Nostalgia", "artist": "Flannel Days", "album": "Seattle Rain", "genre": "Grunge", "mood": "Gritty, 90s, Angsty", "year": "2023", "description": "Heavy, detuned guitars and raw vocals dripping with 90s Seattle grunge attitude and existential angst."},
    {"id": "t054", "title": "Acoustic Campfire", "artist": "Wildwood", "album": "Open Skies", "genre": "Acoustic Rock", "mood": "Warm, Outdoor, Community", "year": "2021", "description": "Warm acoustic guitar strumming with harmonica and group vocals, capturing the magic of singing around a campfire under stars."},
    {"id": "t055", "title": "Electric Storm", "artist": "Voltage Band", "album": "Amplified", "genre": "Hard Rock", "mood": "Intense, Powerful, Electric", "year": "2022", "description": "Face-melting guitar solos and thundering rhythm section creating an electrifying wall of sound that demands headbanging."},

    # ── World / Cultural ───────────────────────────────────────────────────────
    {"id": "t056", "title": "Sahara Sunset", "artist": "Desert Rose", "album": "Oasis", "genre": "World Music", "mood": "Exotic, Warm, Mystical", "year": "2023", "description": "North African rhythms with oud and darbouka percussion, painting golden desert landscapes at the magic hour."},
    {"id": "t057", "title": "Tokyo Neon", "artist": "J-Pop Fusion", "album": "Electric Tokyo", "genre": "J-Pop/Electronic", "mood": "Vibrant, Urban, Japanese", "year": "2022", "description": "High-energy Japanese pop-electronic fusion with sparkling synths and kawaii vocal hooks, capturing Tokyo's electric nightlife."},
    {"id": "t058", "title": "Bollywood Dreams", "artist": "Mumbai Beats", "album": "Rang De", "genre": "Bollywood/Electronic", "mood": "Colorful, Festive, Joyful", "year": "2023", "description": "Bollywood-inspired melodies fused with modern electronic production, like a colorful Holi celebration in sound form."},
    {"id": "t059", "title": "Celtic Highlands", "artist": "Emerald Sound", "album": "Misty Glens", "genre": "Celtic", "mood": "Mystical, Green, Ancient", "year": "2021", "description": "Traditional Celtic instruments including fiddle, tin whistle, and bodhran creating an enchanting journey through misty Scottish highlands."},
    {"id": "t060", "title": "Bossa Nova Nights", "artist": "Rio Rhythm", "album": "Ipanema", "genre": "Bossa Nova", "mood": "Smooth, Tropical, Suave", "year": "2022", "description": "Classic bossa nova rhythm with nylon guitar and smooth Portuguese-style vocals, evoking warm Rio de Janeiro evenings."},

    # ── Gaming / Sci-Fi ────────────────────────────────────────────────────────
    {"id": "t061", "title": "Level Up", "artist": "Pixel Quest", "album": "8-Bit Adventures", "genre": "Chiptune", "mood": "Playful, Retro Gaming, Fun", "year": "2023", "description": "Catchy chiptune melodies with classic 8-bit sound design that transports you straight into a retro arcade game."},
    {"id": "t062", "title": "Cyberpunk City", "artist": "Neon Grid", "album": "2077", "genre": "Cyberpunk", "mood": "Futuristic, Dark, Dystopian", "year": "2022", "description": "Dark, industrial synths with glitchy beats creating a dystopian cyberpunk soundscape of a rain-soaked future megacity."},
    {"id": "t063", "title": "Space Odyssey", "artist": "Cosmos Sound", "album": "Interstellar", "genre": "Space Ambient", "mood": "Vast, Space, Wonder", "year": "2023", "description": "Expansive ambient textures and deep sub-bass representing the infinite vastness and lonely beauty of deep space travel."},
    {"id": "t064", "title": "Boss Fight", "artist": "Game Over", "album": "Final Stage", "genre": "Metal/Electronic", "mood": "Intense, Boss Battle, Epic", "year": "2021", "description": "Heavy metal guitars mixed with electronic elements creating the ultimate boss fight soundtrack with relentless intensity."},
    {"id": "t065", "title": "Quantum Dreams", "artist": "Neural Link", "album": "Singularity", "genre": "IDM", "mood": "Complex, Futuristic, Cerebral", "year": "2022", "description": "Intricate, glitchy electronic patterns and evolving sound design exploring the intersection of artificial intelligence and consciousness."},

    # ── Reggae / Island ─────────────────────────────────────────────────────────
    {"id": "t066", "title": "Island Breeze", "artist": "Jamaican Sunset", "album": "One Love", "genre": "Reggae", "mood": "Relaxed, Island, Positive", "year": "2023", "description": "Classic reggae rhythm with offbeat guitar skanks and warm bass, bringing instant island vibes and positive energy."},
    {"id": "t067", "title": "Dub Meditation", "artist": "Echo Chamber", "album": "Roots & Dub", "genre": "Dub", "mood": "Deep, Meditative, Spacey", "year": "2022", "description": "Heavy dub bass with trippy delay effects and spacious production creating a meditative, head-nodding experience."},
    {"id": "t068", "title": "Tropical Morning", "artist": "Sunrise Beach", "album": "Paradise Found", "genre": "Reggae Pop", "mood": "Happy, Morning, Tropical", "year": "2023", "description": "Upbeat reggae-pop fusion with bright ukulele and steel drums, like waking up to a beautiful tropical sunrise."},

    # ── Jazz ────────────────────────────────────────────────────────────────────
    {"id": "t069", "title": "Smoky Blues", "artist": "Midnight Jazz Club", "album": "Blue Note", "genre": "Jazz Blues", "mood": "Smoky, Cool, Late Night", "year": "2022", "description": "Cool jazz trumpet and walking bass in a smoky club setting, channeling the spirit of classic Blue Note recordings."},
    {"id": "t070", "title": "Swing Time", "artist": "Big Band Revival", "album": "Golden Era", "genre": "Swing", "mood": "Energetic, Vintage, Dancing", "year": "2023", "description": "Full big band arrangement with swinging brass and rhythm section that makes it impossible not to snap your fingers and dance."},

    # ── Country / Americana ────────────────────────────────────────────────────
    {"id": "t071", "title": "Dusty Roads", "artist": "Country Soul", "album": "Heartland", "genre": "Country", "mood": "Rustic, Open Road, Storytelling", "year": "2023", "description": "Twangy guitar and warm vocals telling stories of small-town life, dusty back roads, and the beauty of simple living."},
    {"id": "t072", "title": "Honky Tonk Night", "artist": "Bar Room Band", "album": "Last Call", "genre": "Country", "mood": "Fun, Bar, Rowdy", "year": "2022", "description": "Upbeat honky-tonk with fiddle and pedal steel guitar, capturing a wild Saturday night at a roadside bar."},

    # ── Metal ───────────────────────────────────────────────────────────────────
    {"id": "t073", "title": "Iron Forge", "artist": "Heavy Anvil", "album": "Molten Core", "genre": "Metal", "mood": "Aggressive, Powerful, Dark", "year": "2023", "description": "Crushing downtuned riffs and blast beats forged in pure metal fury, an onslaught of sonic aggression and power."},
    {"id": "t074", "title": "Viking Saga", "artist": "Norse Thunder", "album": "Valhalla", "genre": "Folk Metal", "mood": "Epic, Norse, Warrior", "year": "2022", "description": "Folk metal combining heavy guitars with traditional Norse instruments, telling tales of Viking warriors sailing to glory."},

    # ── Soul / Funk ─────────────────────────────────────────────────────────────
    {"id": "t075", "title": "Groove Machine", "artist": "Funk Dynasty", "album": "Get Down", "genre": "Funk", "mood": "Groovy, Funky, Dancing", "year": "2023", "description": "Tight funk groove with slap bass, wah-wah guitar, and brass stabs that make your body move involuntarily."},
    {"id": "t076", "title": "Soul Kitchen", "artist": "Velvet Voice", "album": "Heart & Soul", "genre": "Soul", "mood": "Warm, Soulful, Uplifting", "year": "2022", "description": "Rich, soulful vocals over warm organ chords and gospel-influenced harmonies celebrating love, life, and resilience."},

    # ── Pop ─────────────────────────────────────────────────────────────────────
    {"id": "t077", "title": "Sunshine Pop", "artist": "Happy Days", "album": "Bright Side", "genre": "Pop", "mood": "Happy, Bright, Uplifting", "year": "2023", "description": "Catchy synth-pop with cheerful melodies and handclaps that radiate happiness and make everything feel bright and possible."},
    {"id": "t078", "title": "Heartbreak Hotel", "artist": "Pop Tears", "album": "Broken Mirror", "genre": "Pop Ballad", "mood": "Heartbroken, Dramatic, Emotional", "year": "2022", "description": "Powerful pop ballad with soaring vocals and emotional piano exploring the devastation and drama of heartbreak."},
    {"id": "t079", "title": "Dance All Night", "artist": "Electro Pop Star", "album": "Neon Pulse", "genre": "Electro Pop", "mood": "Danceable, Fun, Electric", "year": "2023", "description": "High-energy electro-pop banger with infectious hooks and pulsing synths designed for non-stop dancing until sunrise."},
    {"id": "t080", "title": "Teenage Dreams", "artist": "Youth Forever", "album": "Golden Years", "genre": "Indie Pop", "mood": "Youthful, Carefree, Nostalgic", "year": "2021", "description": "Bright indie pop capturing the magic and invincibility of being young, with jangling guitars and carefree melodies."},

    # ── Meditation / Wellness ──────────────────────────────────────────────────
    {"id": "t081", "title": "Chakra Alignment", "artist": "Inner Peace", "album": "Sacred Space", "genre": "Meditation", "mood": "Spiritual, Healing, Centered", "year": "2023", "description": "Tibetan singing bowls and gentle drone tones designed to align energy centers and promote deep healing meditation."},
    {"id": "t082", "title": "Yoga Flow", "artist": "Zen Garden", "album": "Breathe", "genre": "New Age", "mood": "Flowing, Peaceful, Balanced", "year": "2022", "description": "Flowing ambient textures with soft Indian flute perfectly paced for vinyasa yoga practice and mindful breathing."},
    {"id": "t083", "title": "Sleep Waves", "artist": "Dreamland", "album": "Deep Sleep", "genre": "Sleep Music", "mood": "Sleepy, Gentle, Lullaby", "year": "2023", "description": "Ultra-soothing blend of gentle waves and whisper-quiet synth pads that gradually fade, designed to guide you into deep sleep."},

    # ── Blues ───────────────────────────────────────────────────────────────────
    {"id": "t084", "title": "Delta Sunrise", "artist": "Blues Roots", "album": "Mississippi", "genre": "Blues", "mood": "Soulful, Raw, Earthy", "year": "2022", "description": "Raw Delta blues with slide guitar and weathered vocals telling tales of hardship, resilience, and the open Mississippi road."},
    {"id": "t085", "title": "Electric Blues Jam", "artist": "Blues Machine", "album": "Plugged In", "genre": "Electric Blues", "mood": "Gritty, Electric, Jamming", "year": "2023", "description": "Scorching electric guitar solos and driving rhythm section in an extended blues jam that never wants to stop."},

    # ── Electronic / Experimental ──────────────────────────────────────────────
    {"id": "t086", "title": "Glitch Garden", "artist": "Data Bloom", "album": "Corrupted Beauty", "genre": "Glitch", "mood": "Experimental, Digital, Organic", "year": "2023", "description": "Beautiful collisions of organic and digital sounds with glitch textures and evolving ambient garden soundscapes."},
    {"id": "t087", "title": "Techno Cathedral", "artist": "Berlin Nights", "album": "Underground", "genre": "Techno", "mood": "Dark, Hypnotic, Underground", "year": "2022", "description": "Driving Berlin-style techno with hypnotic loops and industrial textures pounding through a cavernous underground warehouse."},
    {"id": "t088", "title": "Trance State", "artist": "Euphoria", "album": "Above & Beyond", "genre": "Trance", "mood": "Euphoric, Transcendent, Uplifting", "year": "2023", "description": "Classic uplifting trance with emotional breakdowns and euphoric climaxes that transport you to a higher state of consciousness."},

    # ── Seasonal / Holiday ─────────────────────────────────────────────────────
    {"id": "t089", "title": "Winter Wonderland", "artist": "Snowfall", "album": "December Magic", "genre": "Holiday", "mood": "Festive, Cozy, Winter", "year": "2022", "description": "Sparkling bells and warm orchestral arrangement evoking the magic of fresh snowfall and cozy holiday evenings by the fireplace."},
    {"id": "t090", "title": "Spring Awakening", "artist": "New Bloom", "album": "Renewal", "genre": "Ambient Pop", "mood": "Fresh, Renewal, Bright", "year": "2023", "description": "Airy pop textures with birdsong samples and bright chords celebrating the joy of spring returning after a long winter."},
    {"id": "t091", "title": "Cafe Paris", "artist": "French Quarter", "album": "Rive Gauche", "genre": "French Jazz", "mood": "Romantic, Parisian, Charming", "year": "2023", "description": "Accordion and gypsy jazz guitar creating the perfect Parisian cafe atmosphere on a warm afternoon by the Seine."},
    {"id": "t092", "title": "Trap Kingdom", "artist": "808 Mafia", "album": "Crown", "genre": "Trap", "mood": "Hard, Dark, Menacing", "year": "2022", "description": "Dark atmospheric trap with thunderous 808 bass and ominous synths building an empire of sonic dominance."},
    {"id": "t093", "title": "Indie Summer", "artist": "Sunflower Band", "album": "Golden Days", "genre": "Indie Rock", "mood": "Summer, Indie, Carefree", "year": "2023", "description": "Sun-drenched indie rock with jangly guitars and sing-along choruses capturing endless summer days and youth."},
    {"id": "t094", "title": "Piano in the Rain", "artist": "Solo Keys", "album": "Raindrops", "genre": "Piano", "mood": "Emotional, Rainy, Solo", "year": "2021", "description": "Solo piano performance with rain ambiance creating an intensely emotional and intimate listening experience."},
    {"id": "t095", "title": "African Drum Circle", "artist": "Tribal Rhythms", "album": "Motherland", "genre": "Afrobeat", "mood": "Rhythmic, Tribal, Energetic", "year": "2022", "description": "Polyrhythmic drum patterns and energetic percussion creating an irresistible groove rooted in West African musical traditions."},
    {"id": "t096", "title": "Dark Ambient Horror", "artist": "Shadow Realm", "album": "Nightmares", "genre": "Dark Ambient", "mood": "Scary, Eerie, Unsettling", "year": "2023", "description": "Deeply unsettling dark ambient soundscapes with distant screams and ominous drones that chill you to the bone."},
    {"id": "t097", "title": "Gospel Morning", "artist": "Praise Choir", "album": "Blessed", "genre": "Gospel", "mood": "Uplifting, Spiritual, Joyful", "year": "2022", "description": "Powerful gospel choir with organ and drums building waves of spiritual joy and uplifting praise that moves the soul."},
    {"id": "t098", "title": "K-Pop Explosion", "artist": "Neon Stars", "album": "Hallyu Wave", "genre": "K-Pop", "mood": "Energetic, Catchy, Polished", "year": "2023", "description": "Slick K-pop production with impossibly catchy hooks, perfectly choreographed breaks, and rapid genre-shifting energy."},
    {"id": "t099", "title": "Acoustic Sunrise", "artist": "Morning Light", "album": "New Day", "genre": "Acoustic", "mood": "Peaceful, Morning, Hopeful", "year": "2021", "description": "Gentle acoustic guitar greeting the sunrise with hopeful melodies that make every new day feel full of possibility."},
    {"id": "t100", "title": "Underground Rave", "artist": "Acid House", "album": "Warehouse", "genre": "Acid House", "mood": "Underground, Acid, Rave", "year": "2022", "description": "Squelchy 303 acid lines and pounding kick drums recreating the raw energy of an illegal warehouse rave at 4 AM."},
]

# ── Preview URL Enrichment ────────────────────────────────────────────────
# Assign stable, royalty-free MP3 URLs to ALL tracks for playback
PREVIEW_URLS = [
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-11.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-12.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-13.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-14.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-15.mp3",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-16.mp3",
]

for i, track in enumerate(MUSIC_DATA):
    track["preview_url"] = PREVIEW_URLS[i % len(PREVIEW_URLS)]


def main():
    print("=" * 60)
    print("  🎵 Music Discovery Engine — Indexer")
    print("=" * 60)

    # ── Load embedding model ───────────────────────────────────────────────
    print("\n📦 Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # ── Connect to Endee ───────────────────────────────────────────────────
    print(f"🔌 Connecting to Endee at {ENDEE_HOST}...")
    client = Endee()

    # ── Create index ───────────────────────────────────────────────────────
    print(f"📁 Creating index '{INDEX_NAME}' (dim={VECTOR_DIMENSION})...")
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
        print(f"   ✅ Index '{INDEX_NAME}' created.")
    except Exception as e:
        print(f"   ℹ️  Index note (may already exist): {e}")

    index = client.get_index(name=INDEX_NAME)

    # ── Generate embeddings and upsert ────────────────────────────────────
    total = len(MUSIC_DATA)
    print(f"\n🎶 Processing {total} tracks...\n")

    for i in range(0, total, BATCH_SIZE):
        batch = MUSIC_DATA[i:i + BATCH_SIZE]

        # Combine all metadata fields into a rich text for embedding
        texts = []
        for track in batch:
            text = (
                f"Title: {track['title']}. "
                f"Artist: {track['artist']}. "
                f"Genre: {track['genre']}. "
                f"Mood: {track['mood']}. "
                f"Description: {track['description']}"
            )
            texts.append(text)

        embeddings = model.encode(texts)
        vectors = []
        for j, track in enumerate(batch):
            # Calculate era
            year_int = int(track["year"])
            era = f"{(year_int // 10) * 10}s"

            vectors.append({
                "id": track["id"],
                "vector": embeddings[j].tolist(),
                "meta": {
                    "title": track["title"],
                    "artist": track["artist"],
                    "album": track["album"],
                    "genre": track["genre"],
                    "mood": track["mood"],
                    "year": track["year"],
                    "era": era,
                    "preview_url": track.get("preview_url", ""),
                    "description": track["description"],
                }
            })

        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   ⬆️  Upserting batch {batch_num}/{total_batches} ({len(batch)} tracks)...")

        try:
            index.upsert(vectors)
        except Exception as e:
            print(f"   ⚠️  Upsert warning: {e}")

    print(f"\n✅ Done! {total} tracks indexed into Endee.")
    print("   You can now start the API server with: python main.py")


if __name__ == "__main__":
    time.sleep(2)  # Give Endee a moment if it just started
    main()
