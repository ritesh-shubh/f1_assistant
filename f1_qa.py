"""
F1 Q&A Pipeline
---------------
Pattern matching engine answers questions directly from your CSVs.
LLM (Ollama) is only used to format the final answer as a natural sentence.
No hallucinations — data answers everything, LLM only formats.

Requirements:
    pip install pandas requests
    Install Ollama from ollama.com and run: ollama pull deepseek-coder

Usage:
    python f1_qa.py              # interactive CLI
    python f1_qa.py "question"   # single question
"""

import os, sys, re
import pandas as pd
import requests

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-coder"
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    print("Loading F1 data...")
    dfs = {
        "race_summaries":        pd.read_csv(f"{DATA_DIR}/race_summaries.csv"),
        "race_details":          pd.read_csv(f"{DATA_DIR}/race_details.csv"),
        "driver_standings":      pd.read_csv(f"{DATA_DIR}/driver_standings.csv"),
        "driver_details":        pd.read_csv(f"{DATA_DIR}/driver_details.csv"),
        "constructor_standings": pd.read_csv(f"{DATA_DIR}/constructor_standings.csv"),
        "fastest_laps":          pd.read_csv(f"{DATA_DIR}/fastest_laps.csv"),
        "fastestlaps_detailed":  pd.read_csv(f"{DATA_DIR}/fastestlaps_detailed.csv"),
        "pitstops":              pd.read_csv(f"{DATA_DIR}/pitstops.csv"),
        "practices":             pd.read_csv(f"{DATA_DIR}/practices.csv"),
        "qualifyings":           pd.read_csv(f"{DATA_DIR}/qualifyings.csv"),
        "starting_grids":        pd.read_csv(f"{DATA_DIR}/starting_grids.csv"),
        "sprint_results":        pd.read_csv(f"{DATA_DIR}/sprint_results.csv"),
        "sprint_grid":           pd.read_csv(f"{DATA_DIR}/sprint_grid.csv"),
        "team_details":          pd.read_csv(f"{DATA_DIR}/team_details.csv"),
    }
    print(f"✓ Loaded {len(dfs)} datasets\n")
    return dfs


# ─────────────────────────────────────────────
# 2. EXTRACTION HELPERS
# ─────────────────────────────────────────────

def extract_year(q):
    m = re.search(r'\b(19[5-9]\d|200\d|201\d|202[0-2])\b', q)
    return int(m.group()) if m else None

def extract_year_range(q):
    """Extract a year range like 'from 2016 to 2020' or 'between 2010 and 2015'."""
    m = re.search(r'(?:from|between)\s+(19[5-9]\d|200\d|201\d|202[0-2])\s+(?:to|and)\s+(19[5-9]\d|200\d|201\d|202[0-2])', q)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def extract_decade(q):
    m = re.search(r'\b(19[5-9]0s|200[0-9]s|2010s|2020s)\b', q)
    if m:
        s = int(m.group()[:4])
        return s, s + 9, m.group()
    return None, None, None

def extract_drivers(q):
    known = {
        'hamilton':    'Lewis Hamilton',
        'verstappen':  'Max Verstappen',
        'schumacher':  'Michael Schumacher',
        'senna':       'Ayrton Senna',
        'prost':       'Alain Prost',
        'vettel':      'Sebastian Vettel',
        'alonso':      'Fernando Alonso',
        'raikkonen':   'Kimi Räikkönen',
        'räikkönen':   'Kimi Räikkönen',
        'rosberg':     'Nico Rosberg',
        'button':      'Jenson Button',
        'leclerc':     'Charles Leclerc',
        'sainz':       'Carlos Sainz',
        'norris':      'Lando Norris',
        'russell':     'George Russell',
        'perez':       'Sergio Perez',
        'bottas':      'Valtteri Bottas',
        'lauda':       'Niki Lauda',
        'fangio':      'Juan Manuel Fangio',
        'clark':       'Jim Clark',
        'hill':        'Damon Hill',
        'mansell':     'Nigel Mansell',
        'piquet':      'Nelson Piquet',
        'brabham':     'Jack Brabham',
        'stewart':     'Jackie Stewart',
        'farina':      'Nino Farina',
        'ascari':      'Alberto Ascari',
        'hakkinen':    'Mika Häkkinen',
        'häkkinen':    'Mika Häkkinen',
        'villeneuve':  'Jacques Villeneuve',
        'gasly':       'Pierre Gasly',
        'ocon':        'Esteban Ocon',
        'ricciardo':   'Daniel Ricciardo',
        'webber':      'Mark Webber',
        'coulthard':   'David Coulthard',
        'montoya':     'Juan Pablo Montoya',
        'massa':       'Felipe Massa',
        'barrichello': 'Rubens Barrichello',
        'irvine':      'Eddie Irvine',
        'patrese':     'Riccardo Patrese',
        'berger':      'Gerhard Berger',
        # First-name shortcuts
        'lewis':       'Lewis Hamilton',
        'max':         'Max Verstappen',
        'charles':     'Charles Leclerc',
        'carlos':      'Carlos Sainz',
        'lando':       'Lando Norris',
        'george':      'George Russell',
        'sergio':      'Sergio Perez',
        'checo':       'Sergio Perez',
        'ayrton':      'Ayrton Senna',
        'michael':     'Michael Schumacher',
        'sebastian':   'Sebastian Vettel',
        'fernando':    'Fernando Alonso',
        'nico':        'Nico Rosberg',
        'jenson':      'Jenson Button',
        'valtteri':    'Valtteri Bottas',
        'kimi':        'Kimi Räikkönen',
        'damon':       'Damon Hill',
        'graham':      'Graham Hill',
        'nigel':       'Nigel Mansell',
        'alain':       'Alain Prost',
        'mika':        'Mika Häkkinen',
        'daniel':      'Daniel Ricciardo',
        'pierre':      'Pierre Gasly',
        'esteban':     'Esteban Ocon',
        'mark':        'Mark Webber',
    }
    found = []
    for key, full in known.items():
        if re.search(r'\b' + re.escape(key) + r'\b', q):
            if full not in found:
                found.append(full)
    return found

def extract_team(q):
    known = [
        ('red bull',    'Red Bull'),
        ('ferrari',     'Ferrari'),
        ('mercedes',    'Mercedes'),
        ('mclaren',     'McLaren'),
        ('williams',    'Williams'),
        ('renault',     'Renault'),
        ('lotus',       'Lotus'),
        ('brawn',       'Brawn'),
        ('benetton',    'Benetton'),
        ('alphatauri',  'AlphaTauri'),
        ('alpine',      'Alpine'),
        ('haas',        'Haas'),
        ('aston martin','Aston Martin'),
        ('alfa romeo',  'Alfa Romeo'),
        ('cooper',      'Cooper'),
        ('tyrrell',     'Tyrrell'),
        ('ligier',      'Ligier'),
        ('brabham',     'Brabham'),
    ]
    for key, full in known:
        if key in q:
            return full
    return None

def extract_gp(q, race_summaries):
    gps = sorted(race_summaries['Grand Prix'].unique(), key=len, reverse=True)
    for gp in gps:
        if gp.lower() in q:
            return gp
    aliases = {
        'british':      'Great Britain',
        'britain':      'Great Britain',
        'usa':          'United States',
        'us gp':        'United States',
        'emilia':       'Emilia Romagna',
        'imola':        'Emilia Romagna',
        '70th':         '70th Anniversary',
        'san marino':   'Emilia Romagna',
        'italian':      'Italy',
        'silverstone':  'Great Britain',
        'monza':        'Italy',
        'suzuka':       'Japan',
        'japanese':     'Japan',
        'belgian':      'Belgium',
        'spa':          'Belgium',
        'german':       'Germany',
        'hungarian':    'Hungary',
        'hungarian':    'Hungary',
        'australian':   'Australia',
        'melbourne':    'Australia',
        'bahrain':      'Bahrain',
        'spanish':      'Spain',
        'canadian':     'Canada',
        'french':       'France',
        'dutch':        'Netherlands',
        'zandvoort':    'Netherlands',
        'portuguese':   'Portugal',
        'mexican':      'Mexico',
        'brazilian':    'Brazil',
        'interlagos':   'Brazil',
        'turkish':      'Turkey',
        'chinese':      'China',
        'singapore':    'Singapore',
        'russian':      'Russia',
        'sochi':        'Russia',
        'azerbaijani':  'Azerbaijan',
        'baku':         'Azerbaijan',
        'saudi':        'Saudi Arabia',
        'abu dhabi':    'Abu Dhabi',
        'qatar':        'Qatar',
        'miami':        'Miami',
    }
    for alias, gp in aliases.items():
        if alias in q:
            return gp
    return None


# ─────────────────────────────────────────────
# 3. PATTERN MATCHING QUERY ENGINE
#    Every branch tested against real data.
# ─────────────────────────────────────────────

def query_data(question, dfs):
    q = question.lower().strip().rstrip('?').strip()

    rs = dfs['race_summaries']
    rd = dfs['race_details']
    ds = dfs['driver_standings']
    cs = dfs['constructor_standings']
    fl = dfs['fastest_laps']
    ps = dfs['pitstops']
    sg = dfs['starting_grids']
    qu = dfs['qualifyings']
    dd = dfs['driver_details']

    year          = extract_year(q)
    yr_start, yr_end = extract_year_range(q)
    dec_s, dec_e, dec_str = extract_decade(q)
    drivers       = extract_drivers(q)
    team          = extract_team(q)
    gp            = extract_gp(q, rs)

    # ─────────────────────────────────────────────────────────────────────
    # Helper: resolve full driver name to exact match in dataframes
    # ─────────────────────────────────────────────────────────────────────
    def dmatch(df, col, driver_name):
        """Match driver by full name if possible, else last name word-boundary."""
        parts = driver_name.split()
        if len(parts) >= 2:
            rows = df[df[col] == driver_name]
            if len(rows): return rows
        last = parts[-1]
        return df[df[col].str.contains(r'\b' + re.escape(last) + r'\b', case=False, na=False, regex=True)]

    # ── 1. DECADE WINS ────────────────────────────────────────────────────
    if dec_str and any(w in q for w in ['win','wins','won','race','races']):
        if team:
            n = rs[(rs['Year']>=dec_s)&(rs['Year']<=dec_e)&rs['Car'].str.contains(team,case=False,na=False)].shape[0]
            return "{} won {} races in the {}.".format(team, n, dec_str)
        if drivers:
            d=drivers[0]; rows=dmatch(rs,'Winner',d)
            rows=rows[(rows['Year']>=dec_s)&(rows['Year']<=dec_e)]
            return "{} won {} races in the {}.".format(d, len(rows), dec_str)
        top = rs[(rs['Year']>=dec_s)&(rs['Year']<=dec_e)].groupby('Winner').size().sort_values(ascending=False)
        return "Most wins in the {}: {} with {}.".format(dec_str, top.index[0], top.iloc[0])

    # ── 2. WINS FROM POLE ─────────────────────────────────────────────────
    if 'from pole' in q:
        merged = rs.merge(sg[sg['Pos']==1][['Year','Grand Prix','Driver']], on=['Year','Grand Prix'])
        if drivers:
            d=drivers[0]; rows=dmatch(merged,'Driver',d)
            return "{} has won {} races from pole position.".format(d, len(rows))
        top = merged.groupby('Driver').size().sort_values(ascending=False)
        return "{} has won the most races from pole with {}.".format(top.index[0], top.iloc[0])

    # ── 2a. WINS FROM LOW GRID POSITION ─────────────────────────────────
    if any(w in q for w in ['starting from','won from','won a race starting','won starting','won a race from','won from position']):
        m_pos=re.search(r'(\d+)(?:th|st|nd|rd)?',q)
        threshold=int(m_pos.group(1)) if m_pos else 10
        merged=rs.merge(sg,left_on=['Year','Grand Prix','Winner'],right_on=['Year','Grand Prix','Driver'],how='inner')
        merged['Pos']=pd.to_numeric(merged['Pos'],errors='coerce')
        low_start_wins=merged[merged['Pos']>=threshold].sort_values('Pos',ascending=False)
        if len(low_start_wins):
            top=low_start_wins.iloc[0]
            count=len(low_start_wins)
            return "{} races have been won by drivers starting P{} or lower. The lowest grid start for a winner was P{} by {} at the {} {} Grand Prix.".format(
                count,threshold,int(top['Pos']),top['Winner'],int(top['Year']),top['Grand Prix'])
        return "No driver has won a race starting from P{} or lower in this dataset.".format(threshold)

    # ── 3. WHICH NATIONALITY HAS WON MOST RACES ──────────────────────────
    if any(w in q for w in ['which nationality','what nationality has','nationality has won','nationality won most']):
        nat_wins = rs.merge(ds[['Driver','Nationality']].drop_duplicates(), left_on='Winner', right_on='Driver', how='left')
        top = nat_wins.groupby('Nationality').size().sort_values(ascending=False)
        nat_map = {'GBR':'British','GER':'German','BRA':'Brazilian','FIN':'Finnish','FRA':'French',
                   'AUS':'Australian','AUT':'Austrian','ITA':'Italian','NED':'Dutch','ESP':'Spanish',
                   'CAN':'Canadian','BEL':'Belgian','ARG':'Argentine','SUI':'Swiss'}
        top5 = "\n".join("  {} ({}): {} wins".format(nat_map.get(k,k), k, v) for k,v in top.head(5).items())
        return "Race wins by nationality:\n{}".format(top5)

    # ── 4. DRIVER NATIONALITY ─────────────────────────────────────────────
    if any(w in q for w in ['nationality','what country','where is','where was','born in']) and len(drivers)==1:
        d=drivers[0]; rows=dmatch(ds,'Driver',d)
        if len(rows):
            code=rows.iloc[0]['Nationality']
            nat_map={'GBR':'British','GER':'German','BRA':'Brazilian','FIN':'Finnish','FRA':'French',
                     'AUS':'Australian','AUT':'Austrian','ITA':'Italian','NED':'Dutch','ESP':'Spanish',
                     'CAN':'Canadian','BEL':'Belgian','ARG':'Argentine','NZL':'New Zealander',
                     'SCO':'Scottish','SWE':'Swedish','USA':'American','RSA':'South African',
                     'MEX':'Mexican','COL':'Colombian','VEN':'Venezuelan','DEN':'Danish',
                     'MON':'Monégasque','POL':'Polish','RUS':'Russian','JPN':'Japanese',
                     'CHN':'Chinese','IND':'Indian','THA':'Thai','CHI':'Chilean',
                     'SUI':'Swiss','HUN':'Hungarian','IRL':'Irish','POR':'Portuguese'}
            return "{} is {} ({}).".format(d, nat_map.get(code,code), code)

    # ── 5. DRIVER DEBUT ───────────────────────────────────────────────────
    # Must come BEFORE "first ever F1 race" block
    if len(drivers)==1 and any(w in q for w in ['debut','first season','first year','started racing',
                                                  'began racing','first raced','first drove','first race','what year did']):
        d=drivers[0]
        # Use race_details for debut as it has more complete records
        rows_rd = dmatch(rd,'Driver',d)
        if len(rows_rd):
            first_yr = rows_rd['Year'].min()
            # Get team from driver_standings that year
            rows_ds = dmatch(ds,'Driver',d)
            rows_ds = rows_ds[rows_ds['Year']==first_yr]
            team_name = rows_ds.iloc[0]['Car'] if len(rows_ds) else 'unknown team'
            return "{} made their F1 debut in {} driving for {}.".format(d, first_yr, team_name)

    # ── 6. DRIVER RETIREMENT / LAST SEASON ───────────────────────────────
    if len(drivers)==1 and any(w in q for w in ['retire','retired','retirement','last season racing',
                                                  'stopped racing','final season','last year racing']):
        d=drivers[0]; rows=dmatch(ds,'Driver',d)
        if len(rows):
            final_yr=rows['Year'].max(); first_yr=rows['Year'].min(); seasons=rows['Year'].nunique()
            if final_yr < 2022:
                return "{} last raced in F1 in {} (debut: {}, {} seasons in this dataset).".format(d, final_yr, first_yr, seasons)
            return "{} was still active in F1 in 2022 (debut: {}, {} seasons — data ends 2022).".format(d, first_yr, seasons)

    # ── 7. HOW MANY SEASONS RACED ─────────────────────────────────────────
    if len(drivers)==1 and any(w in q for w in ['how many years','how many seasons','seasons in f1','years in f1']):
        d=drivers[0]; rows=dmatch(ds,'Driver',d)
        if len(rows):
            return "{} competed in {} F1 seasons ({}-{}) in this dataset.".format(
                d, rows['Year'].nunique(), rows['Year'].min(), rows['Year'].max())

    # ── 8. IS DRIVER STILL RACING ─────────────────────────────────────────
    if any(w in q for w in ['still racing','still driving','still active','still in f1','is he still']) and len(drivers)==1:
        d=drivers[0]; rows=dmatch(ds,'Driver',d)
        if len(rows):
            last_yr=rows['Year'].max()
            if last_yr==2022:
                return "{} was racing in 2022 (the final year in this dataset). Check current sources for 2023+.".format(d)
            return "{} last appeared in the F1 standings in {}. They may have retired (data ends 2022).".format(d, last_yr)

    # ── 9. WHO DROVE FOR A TEAM ───────────────────────────────────────────
    if any(w in q for w in ['who drives','who drove','who race','which drivers','who were','lineup','team drivers','who are the drivers']) and team:
        yr = year if year else ds[ds['Car'].str.contains(team,case=False,na=False)]['Year'].max()
        rows = ds[(ds['Year']==yr) & ds['Car'].str.lower().str.startswith(team.lower())]
        if not len(rows):
            rows = ds[(ds['Year']==yr) & ds['Car'].str.contains(team,case=False,na=False)]
        main = list(rows['Driver'].unique()[:2])
        if main:
            return "{} drivers in {}: {}.".format(team, yr, ' and '.join(main))

    # ── 10. HOW DID A TEAM DO IN A SEASON ────────────────────────────────
    if any(w in q for w in ['how did','team do','team perform']) and team and year:
        c_rows=cs[(cs['Year']==year) & cs['Team'].str.contains(team,case=False,na=False)]
        wins=rs[(rs['Year']==year) & rs['Car'].str.contains(team,case=False,na=False)].shape[0]
        if len(c_rows):
            r=c_rows.iloc[0]
            return "{} finished P{} in the {} Constructors Championship with {:.0f} points and {} race win{}.".format(
                r['Team'], r['Pos'], year, r['PTS'], wins, 's' if wins!=1 else '')

    # ── 11. MOST WINS IN A SINGLE SEASON ─────────────────────────────────
    if any(w in q for w in ['most wins in a season','most wins in one season','most wins in a single',
                             'record wins in a season','wins in a single season','most races won in a season']):
        top=rs.groupby(['Year','Winner']).size().reset_index(name='W').sort_values('W',ascending=False).iloc[0]
        return "{} holds the record with {} wins in the {} season.".format(top['Winner'], top['W'], int(top['Year']))

    # ── 11a. WINNING MARGIN ──────────────────────────────────────────────
    if any(w in q for w in ['winning margin','biggest margin','largest margin','biggest gap','largest gap','margin of victory']):
        try:
            p2=rd[pd.to_numeric(rd['Pos'],errors='coerce')==2].copy()
            p2['Gap']=pd.to_numeric(p2['Time/Retired'].astype(str).str.replace('+','',regex=False).str.replace('s','',regex=False).str.strip(),errors='coerce')
            valid=p2[p2['Gap']>0].dropna(subset=['Gap'])
            if len(valid):
                best=valid.loc[valid['Gap'].idxmax()]
                winner_row=rs[(rs['Year']==int(best['Year']))&rs['Grand Prix'].str.contains(str(best['Grand Prix']),case=False,na=False)]
                winner=winner_row.iloc[0]['Winner'] if len(winner_row) else 'Unknown'
                return "The biggest winning margin (by time gap) was {:.3f}s at the {} {} Grand Prix, won by {}.".format(
                    best['Gap'],int(best['Year']),best['Grand Prix'],winner)
        except (KeyError, ValueError, TypeError):
            pass
        return "Winning margin data is not directly available in this dataset."

    # ── 12. MOST POINTS IN A SEASON ──────────────────────────────────────
    if any(w in q for w in ['most points in a season','most points ever in a season','highest points in a season',
                             'record points in a season','most points scored in a season']):
        top=ds.sort_values('PTS',ascending=False).iloc[0]
        return "{} scored the most points in a single season: {:.0f} points in {}.".format(top['Driver'], top['PTS'], top['Year'])

    # ── 13. LONGEST WINNING STREAK ────────────────────────────────────────
    if not gp and any(w in q for w in ['longest winning streak','consecutive wins','consecutive victories','winning streak','most consecutive','in a row','races in a row']):
        seq=rs.sort_values(['Year','Grand Prix'])['Winner'].tolist()
        max_s=cur=1; name=prev=seq[0]
        for w in seq[1:]:
            cur=cur+1 if w==prev else 1
            if cur>max_s: max_s=cur; name=w
            prev=w
        m_n=re.search(r'(?:more than|over|at least|exceed)\s+(\d+)',q)
        if m_n:
            threshold=int(m_n.group(1))
            if max_s>threshold:
                return "Yes — {} won {} consecutive races, which is more than {}.".format(name, max_s, threshold)
            return "No — the longest winning streak is {} consecutive wins by {}, which is not more than {}.".format(max_s, name, threshold)
        return "{} holds the record with {} consecutive race wins.".format(name, max_s)

    # ── 14. HEAD TO HEAD ─────────────────────────────────────────────────
    if len(drivers)>=2 and any(w in q for w in ['head to head','better','comparison','compare']):
        lines=[]
        for d in drivers[:3]:
            rows_w=dmatch(rs,'Winner',d); wins=len(rows_w)
            rows_sg=dmatch(sg,'Driver',d); poles=len(rows_sg[rows_sg['Pos']==1])
            rows_rd2=dmatch(rd,'Driver',d); podiums=len(rows_rd2[pd.to_numeric(rows_rd2['Pos'],errors='coerce')<=3])
            rows_ds2=dmatch(ds,'Driver',d); champs=len(rows_ds2[rows_ds2['Pos']=='1'])
            lines.append("  {:<26} {:>4} wins | {:>4} poles | {:>4} podiums | {:>2} titles".format(d,wins,poles,podiums,champs))
        return "Head-to-head comparison:\n"+"\n".join(lines)

    # ── 15. WHICH TEAM HAS THE MOST ALL-TIME WINS ─────────────────────────
    if any(w in q for w in ['which team has the most wins','team with the most wins','team with most wins',
                             'most wins by a team','most wins team','most race wins ever','most wins as a constructor']):
        top=rs.groupby('Car').size().sort_values(ascending=False).head(5)
        lines="\n".join("  {}: {}".format(n,c) for n,c in top.items())
        return "Teams with the most F1 wins (all time):\n{}".format(lines)

    # ── 15a. TEAM WIN PERCENTAGE ─────────────────────────────────────────
    if any(w in q for w in ['win%','win percentage','win rate','highest win rate','best win rate','highest win%','dominant','domination','dominance']):
        team_year=rs.groupby(['Year','Car']).size().reset_index(name='Wins')
        races_per_year=rs.groupby('Year').size().reset_index(name='TotalRaces')
        team_year=team_year.merge(races_per_year,on='Year')
        team_year['WinPct']=team_year['Wins']/team_year['TotalRaces']*100
        best=team_year.loc[team_year['WinPct'].idxmax()]
        return "{} had the highest win rate in {} — winning {} of {} races ({:.1f}%).".format(
            best['Car'],int(best['Year']),int(best['Wins']),int(best['TotalRaces']),best['WinPct'])

    # ── 16. HOW MANY TIMES HAS A GP BEEN HELD ────────────────────────────
    if any(w in q for w in ['how many times has','how often has','how many editions','how many times the','how many times a']) and gp:
        sub=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)]
        return "The {} Grand Prix has been held {} times in F1 ({}-{}).".format(gp, len(sub), sub['Year'].min(), sub['Year'].max())

    # ── 17. FIRST [SPECIFIC GP] – before generic "first ever" ────────────
    if 'first' in q and gp and not year and any(w in q for w in ['grand prix','gp','race','win','won','take place','took place','held','when','date','who']):
        rows=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)].sort_values('Year')
        if len(rows):
            r=rows.iloc[0]
            return "The first {} Grand Prix was held in {} and was won by {} driving for {}.".format(gp, r['Year'], r['Winner'], r['Car'])

    # ── 18. FIRST RACE OF A SEASON ────────────────────────────────────────
    if any(w in q for w in ['first race of','opening race','season opener','first grand prix of']) and year:
        r=rs[rs['Year']==year].iloc[0]
        return "The first race of {} was the {} Grand Prix, won by {} ({}).".format(year, r['Grand Prix'], r['Winner'], r['Car'])

    # ── 19a. LAST TO FIRST / WON FROM THE BACK ──────────────────────────
    if any(w in q for w in ['last to first','lsat to first','from last','from the back','back of the grid','last on the grid','from dead last']):
        merged=rs.merge(sg,left_on=['Year','Grand Prix','Winner'],right_on=['Year','Grand Prix','Driver'],how='inner')
        merged['Pos']=pd.to_numeric(merged['Pos'],errors='coerce')
        merged=merged.dropna(subset=['Pos'])
        if len(merged):
            worst_start=merged.sort_values('Pos',ascending=False).iloc[0]
            return "{} won the {} {} Grand Prix starting from P{} — the lowest grid position for a race winner in this dataset.".format(
                worst_start['Winner'],int(worst_start['Year']),worst_start['Grand Prix'],int(worst_start['Pos']))

    # ── 19. FIRST EVER F1 RACE – fires ONLY when no driver/GP/year context ─
    if 'first' in q and not gp and not year and not drivers and any(w in q for w in ['ever','f1','formula 1','formula one']):
        r=rs.sort_values('Year').iloc[0]
        return "The first ever F1 race was won by {} ({}) at the {} {} Grand Prix.".format(r['Winner'], r['Car'], r['Year'], r['Grand Prix'])

    # ── 20a. CONSECUTIVE WINS AT SPECIFIC GP ─────────────────────────────
    if gp and any(w in q for w in ['in a row','consecutive','back to back','twice in a row','three in a row','row at']):
        sub=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)].sort_values('Year')
        if len(sub)>1:
            max_s=cur=1; name=prev=sub.iloc[0]['Winner']
            for _,r in sub.iloc[1:].iterrows():
                w=r['Winner']
                cur=cur+1 if w==prev else 1
                if cur>max_s: max_s=cur; name=w
                prev=w
            threshold=None
            m_n=re.search(r'(?:more than|over|at least)\s+(\d+)',q)
            if m_n: threshold=int(m_n.group(1))
            if 'twice' in q: threshold=2
            if threshold:
                if max_s>=threshold:
                    return "Yes — {} won the {} Grand Prix {} times in a row.".format(name,gp,max_s)
                return "No — the longest consecutive win streak at the {} Grand Prix is {} by {}.".format(gp,max_s,name)
            return "{} holds the record with {} consecutive wins at the {} Grand Prix.".format(name,max_s,gp)

    # ── 20. DRIVER WINS AT SPECIFIC GP ───────────────────────────────────
    if gp and len(drivers)==1 and 'podium' not in q and (any(w in q for w in ['when did','when has','how many times','how often',
        'times did','times has','has','have','ever won','ever win','did win']) or q.split()[0]=='did'
        or ('how many' in q and any(w in q for w in ['win','wins','won']))):
        d=drivers[0]; rows=dmatch(rs,'Winner',d)
        rows=rows[rows['Grand Prix'].str.contains(gp,case=False,na=False)].sort_values('Year')
        if len(rows):
            return "{} won the {} Grand Prix {} time{} ({}).".format(d, gp, len(rows), 's' if len(rows)!=1 else '', ', '.join(str(y) for y in rows['Year']))
        return "{} has never won the {} Grand Prix in this dataset (1950-2022).".format(d, gp)

    # ── 20c. ONLY DRIVER/TEAM TO WIN A SPECIFIC GP ──────────────────────
    # e.g. "who is the only driver to win the miami grand prix"
    if gp and any(w in q for w in ['only driver','only team','only person','only one to']):
        if 'team' in q or 'constructor' in q:
            winners=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)].groupby('Car').size()
            if len(winners)==1:
                return "{} is the only team to have won the {} Grand Prix.".format(winners.index[0],gp)
            return "Multiple teams have won the {} Grand Prix ({} different teams).".format(gp,len(winners))
        winners=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)].groupby('Winner').size()
        if len(winners)==1:
            return "{} is the only driver to have won the {} Grand Prix.".format(winners.index[0],gp)
        return "Multiple drivers have won the {} Grand Prix ({} different winners).".format(gp,len(winners))

    # ── 20d. WON BOTH X AND Y GPs ────────────────────────────────────────
    # e.g. "who has won both the monaco and british grand prix"
    if 'both' in q and 'and' in q and any(w in q for w in ['won both','won the','winner of both']):
        parts=q.split(' and ')
        gp_names=[]
        for part in parts:
            g=extract_gp(part.strip(), rs)
            if g and g not in gp_names: gp_names.append(g)
        if len(gp_names)>=2:
            winner_sets=[set(rs[rs['Grand Prix'].str.contains(g,case=False,na=False)]['Winner'].unique()) for g in gp_names]
            common=winner_sets[0]
            for s in winner_sets[1:]: common=common&s
            common=sorted(common)
            if common:
                lines="\n".join("  {}".format(d) for d in common[:15])
                return "Drivers who have won both the {} Grand Prix:\n{}".format(' and '.join(gp_names),lines)
            return "No driver has won both the {} Grand Prix.".format(' and '.join(gp_names))

    # ── 21. MOST WINS AT SPECIFIC GP ─────────────────────────────────────
    if gp and any(w in q for w in ['most','who has won','who won most','which driver']):
        top=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)].groupby('Winner').size().sort_values(ascending=False)
        if len(top):
            lines="\n".join("  {}: {} win{}".format(n,c,'s' if c!=1 else '') for n,c in top.head(5).items())
            return "Most wins at the {} Grand Prix:\n{}".format(gp, lines)

    # ── 21b. NEVER WON CHAMPIONSHIP BUT WON GP ──────────────────────────
    # e.g. "who has never won a championship but has won the monaco grand prix"
    if 'never' in q and any(w in q for w in ['championship','title','wdc']) and any(w in q for w in ['won','win']) and 'most wins' not in q and 'most race wins' not in q:
        champs=set(ds[ds['Pos']=='1']['Driver'].unique())
        if gp:
            gp_winners=set(rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)]['Winner'].unique())
            non_champ_winners=sorted(gp_winners-champs)
            if non_champ_winners:
                lines="\n".join("  {}".format(d) for d in non_champ_winners[:15])
                return "Drivers who won the {} Grand Prix but never won a World Championship:\n{}".format(gp,lines)
            return "All {} Grand Prix winners in this dataset also won a World Championship.".format(gp)
        all_winners=set(rs['Winner'].unique())
        non_champ_winners=sorted(all_winners-champs)
        if non_champ_winners:
            lines="\n".join("  {}".format(d) for d in non_champ_winners[:20])
            return "Drivers who won races but never won a World Championship:\n{}".format(lines)

    # ── 22. RACE WINNER ───────────────────────────────────────────────────
    if any(w in q for w in ['who won','winner of','win the','won the']) and not any(w in q for w in ['most','constructor']):
        if year and gp:
            rows=rs[(rs['Year']==year)&rs['Grand Prix'].str.contains(gp,case=False,na=False)]
            if len(rows): return "{} won the {} {} Grand Prix driving for {}.".format(rows.iloc[0]['Winner'],year,gp,rows.iloc[0]['Car'])
        if year and not gp:
            rows=rs[rs['Year']==year].sort_values('Grand Prix')
            return "Race winners in {}:\n".format(year)+rows[['Grand Prix','Winner','Car']].to_string(index=False)
        if gp and not year:
            asc='first' in q
            rows=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)].sort_values('Year',ascending=asc)
            return "{} winners at {}:\n".format("Oldest" if asc else "Recent", gp)+rows[['Year','Winner','Car']].head(10).to_string(index=False)

    # ── 23. COMPARISON: who has more wins/poles/podiums ───────────────────
    if len(drivers)>=2 and any(w in q for w in ['more wins','more poles','more podiums','more championships',
                                                 'more titles','between','vs','versus','win more','more races won',
                                                 'who has won more','how many more']):
        cat='wins'
        if 'pole' in q: cat='poles'
        elif 'podium' in q: cat='podiums'
        elif 'championship' in q or 'title' in q: cat='titles'
        lines=[]; results={}
        for d in drivers:
            rows_w=dmatch(rs,'Winner',d); w_cnt=len(rows_w)
            rows_sg2=dmatch(sg,'Driver',d); p_cnt=len(rows_sg2[rows_sg2['Pos']==1])
            rows_rd3=dmatch(rd,'Driver',d); pod_cnt=len(rows_rd3[pd.to_numeric(rows_rd3['Pos'],errors='coerce')<=3])
            rows_ds3=dmatch(ds,'Driver',d); t_cnt=len(rows_ds3[rows_ds3['Pos']=='1'])
            val = {'wins':w_cnt,'poles':p_cnt,'podiums':pod_cnt,'titles':t_cnt}[cat]
            results[d]=val
            lines.append("  {}: {} {}".format(d, val, cat))
        leader=max(results, key=results.get)
        return "Comparing {}:\n{}\n→ {} leads.".format(cat, '\n'.join(lines), leader)

    # ── 24. MOST WINS ALL TIME ────────────────────────────────────────────
    if (any(w in q for w in ['most wins','most races won','all time wins','all-time wins','won most races','won the most races'])
        or (re.search(r'top\s*\d+',q) and any(w in q for w in ['win','wins']))
    ) and 'without' not in q and 'but no' not in q and 'but never' not in q:
        m2=re.search(r'top\s*(\d+)',q)
        if m2 or 'top' in q or 'list' in q:
            n=int(m2.group(1)) if m2 else 5
            top=rs.groupby('Winner').size().sort_values(ascending=False).head(n)
            return "Top {} drivers by race wins:\n".format(n)+'\n'.join("  {}. {}: {}".format(i+1,n2,c) for i,(n2,c) in enumerate(top.items()))
        if year and team:
            n=rs[(rs['Year']==year)&rs['Car'].str.contains(team,case=False,na=False)].shape[0]
            return "{} won {} races in {}.".format(team, n, year)
        if year and not drivers:
            top=rs[rs['Year']==year].groupby('Car').size().sort_values(ascending=False)
            return "{} won the most races in {} with {}.".format(top.index[0], year, top.iloc[0])
        if team:
            n=rs[rs['Car'].str.contains(team,case=False,na=False)].shape[0]
            return "{} has won {} races in total.".format(team, n)
        top=rs.groupby('Winner').size().sort_values(ascending=False)
        return "{} has the most race wins of all time with {} wins.".format(top.index[0], top.iloc[0])

    # ── 25. WIN COUNTS ────────────────────────────────────────────────────
    if any(w in q for w in ['how many','total','number of']) and any(w in q for w in ['win','wins','won','victory','victories']) and not any(w in q for w in ['title','titles','championship','championships']) and 'how many teams' not in q:
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(rs,'Winner',d)
            if year: rows=rows[rows['Year']==year]; return "{} won {} race{} in {}.".format(d,len(rows),'s' if len(rows)!=1 else '',year)
            return "{} has won {} races in total.".format(d, len(rows))
        if team:
            sub=rs[rs['Car'].str.contains(team,case=False,na=False)]
            if year: sub=sub[sub['Year']==year]; return "{} won {} race{} in {}.".format(team,len(sub),'s' if len(sub)!=1 else '',year)
            return "{} has won {} races in total.".format(team, len(sub))

    # ── 26. RUNNER UP / SPECIFIC CHAMPIONSHIP POSITION ────────────────────
    # (must be BEFORE driver championships block)
    if year and any(w in q for w in ['runner up','runner-up','who came second','who came third','who came fourth',
                                      'finished second','finished third','finished fourth','finished fifth',
                                      'who finished second','who finished third','second in the','third in the','fourth in the']):
        pos_map={'second':'2','runner up':'2','runner-up':'2','third':'3','fourth':'4','fifth':'5'}
        target='2'
        for word,pos in pos_map.items():
            if word in q: target=pos; break
        rows=ds[(ds['Year']==year)&(ds['Pos']==target)]
        if len(rows):
            return "{} finished P{} in the {} World Championship with {:.0f} points.".format(rows.iloc[0]['Driver'],target,year,rows.iloc[0]['PTS'])

    # ── 26a0. RACE-BY-RACE HEAD-TO-HEAD ──────────────────────────────────
    # e.g. "how many races did rosberg finish ahead of hamilton"
    if len(drivers)==2 and any(w in q for w in ['finish ahead','finished ahead','ahead of','beat','beaten',
                                                 'races did']) and any(w in q for w in ['race','races','finish','finished']) and not any(w in q for w in ['podium','shared','together','championship','title','season']):
        d1,d2=drivers[0],drivers[1]
        last1,last2=d1.split()[-1],d2.split()[-1]
        r1=rd[rd['Driver'].str.contains(r'\b'+re.escape(last1)+r'\b',case=False,na=False,regex=True)][['Year','Grand Prix','Pos','Driver']]
        r2=rd[rd['Driver'].str.contains(r'\b'+re.escape(last2)+r'\b',case=False,na=False,regex=True)][['Year','Grand Prix','Pos','Driver']]
        merged=r1.merge(r2,on=['Year','Grand Prix'],suffixes=('_1','_2'))
        merged['P1']=pd.to_numeric(merged['Pos_1'],errors='coerce')
        merged['P2']=pd.to_numeric(merged['Pos_2'],errors='coerce')
        valid=merged.dropna(subset=['P1','P2'])
        d1_ahead=int((valid['P1']<valid['P2']).sum())
        d2_ahead=int((valid['P2']<valid['P1']).sum())
        ties=len(valid)-d1_ahead-d2_ahead
        return ("Race-by-race head-to-head ({} races where both finished):\n"
                "  {} finished ahead: {} times\n"
                "  {} finished ahead: {} times{}").format(
                    len(valid),d1,d1_ahead,d2,d2_ahead,
                    "\n  Tied: {} times".format(ties) if ties else '')

    # ── 26a. CHAMPIONSHIP HEAD-TO-HEAD ─────────────────────────────────
    if len(drivers)==2 and any(w in q for w in ['ahead','beat','beaten','finished ahead','ended ahead',
                                                  'finished higher','higher in the championship',
                                                  'more championships than','times max','times hamilton']) and not any(w in q for w in ['podium','shared','together']):
        d1,d2=drivers[0],drivers[1]
        sub=ds[ds['Driver'].isin([d1,d2])]
        piv=sub.pivot_table(index='Year',columns='Driver',values='PTS').dropna()
        if d1 in piv.columns and d2 in piv.columns:
            d1_ahead=int((piv[d1]>piv[d2]).sum()); d2_ahead=int((piv[d2]>piv[d1]).sum())
            n=len(piv)
            return "In {} seasons where both {} and {} competed:\n  {} finished higher: {} times\n  {} finished higher: {} times".format(n,d1,d2,d1,d1_ahead,d2,d2_ahead)

    # ── 26b. YOUNGEST WORLD CHAMPION ────────────────────────────────────
    if any(w in q for w in ['youngest champion','youngest world champion','youngest driver to win','youngest title']):
        return ("Sebastian Vettel became the youngest F1 World Champion in 2010 at age 23. "
                "(Note: this dataset does not include driver birthdays — this is from general F1 knowledge.)")

    # ── 26c. CHAMPIONSHIP WITHOUT WINNING A RACE ────────────────────────
    # e.g. "has anyone won the championship without winning a race"
    if any(w in q for w in ['championship without','title without','champion without']) and any(w in q for w in ['winning a race','race win','winning any race','a win','any wins']):
        champ_years=ds[ds['Pos']=='1'][['Driver','Year']]
        results=[]
        for _,row in champ_years.iterrows():
            wins=rs[(rs['Year']==row['Year'])&(rs['Winner']==row['Driver'])].shape[0]
            if wins==0:
                results.append("{} in {}".format(row['Driver'],row['Year']))
        if results:
            return "Drivers who won the championship without winning a race that season:\n{}".format("\n".join("  "+r for r in results))
        return "Every World Champion in this dataset won at least one race in their title-winning season."

    # ── 26d. MOST WINS WITHOUT A CHAMPIONSHIP ─────────────────────────────
    # e.g. "who has the most wins without winning the championship"
    if any(w in q for w in ['most wins without','wins without a championship','wins without a title',
                             'wins without winning the championship','wins without winning a title',
                             'most race wins without','wins but no championship','wins but never won the championship']):
        champs=set(ds[ds['Pos']=='1']['Driver'].unique())
        win_counts=rs.groupby('Winner').size().sort_values(ascending=False)
        non_champ_wins=win_counts[~win_counts.index.isin(champs)]
        if len(non_champ_wins):
            lines="\n".join("  {}: {} win{}".format(d,c,'s' if c!=1 else '') for d,c in non_champ_wins.head(5).items())
            return "Most race wins without ever winning a World Championship:\n{}".format(lines)

    # ── 27. DRIVER CHAMPIONSHIPS ──────────────────────────────────────────
    if any(w in q for w in ['championship','world champion','title','wdc']) and not any(w in q for w in ['constructor','team','wcc']) and not team:
        if year:
            rows=ds[(ds['Year']==year)&(ds['Pos']=='1')]
            if len(rows): return "{} won the {} World Drivers Championship.".format(rows.iloc[0]['Driver'], year)
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(ds,'Driver',d); n=len(rows[rows['Pos']=='1'])
            return "{} has won {} World Drivers Championship{}.".format(d, n, 's' if n!=1 else '')
        top=ds[ds['Pos']=='1'].groupby('Driver').size().sort_values(ascending=False)
        return "{} has won the most driver championships with {} titles.".format(top.index[0], top.iloc[0])

    # ── 28. CONSTRUCTOR CHAMPIONSHIPS ────────────────────────────────────
    if any(w in q for w in ['constructor','constructors','wcc','team championship']) or (team and any(w in q for w in ['championship','championships','title','titles'])):
        if year:
            rows=cs[(cs['Year']==year)&(cs['Pos']=='1')]
            if len(rows): return "{} won the {} Constructors Championship.".format(rows.iloc[0]['Team'], year)
        if team:
            # Use startswith to avoid "McLaren Mercedes" counting for "Mercedes"
            rows=cs[cs['Team'].str.lower().str.startswith(team.lower())&(cs['Pos']=='1')]
            if not len(rows):
                rows=cs[cs['Team'].str.contains(team,case=False,na=False)&(cs['Pos']=='1')]
            n=len(rows)
            return "{} has won {} Constructors Championship{}.".format(team, n, 's' if n!=1 else '')
        top=cs[cs['Pos']=='1'].groupby('Team').size().sort_values(ascending=False)
        return "{} has won the most Constructors Championships with {} titles.".format(top.index[0], top.iloc[0])

    # ── 29. FASTEST LAPS ──────────────────────────────────────────────────
    if any(w in q for w in ['fastest lap','fastest laps']):
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(fl,'Driver',d)
            return "{} has set {} fastest laps in their career.".format(d, len(rows))
        if year and gp:
            rows=fl[(fl['Year']==year)&fl['Grand Prix'].str.contains(gp,case=False,na=False)]
            if len(rows): return "{} set the fastest lap at the {} {} Grand Prix ({}).".format(rows.iloc[0]['Driver'],year,gp,rows.iloc[0]['Time'])
        top=fl.groupby('Driver').size().sort_values(ascending=False)
        m_n=re.search(r'top\s*(\d+)',q)
        if m_n or 'top' in q or 'list' in q or 'rank' in q:
            n=int(m_n.group(1)) if m_n else 10
            return "Top {} drivers by fastest laps:\n".format(n)+'\n'.join("  {}. {}: {}".format(i+1,d,c) for i,(d,c) in enumerate(top.head(n).items()))
        return "{} holds the record for most fastest laps with {}.".format(top.index[0], top.iloc[0])

    # ── 30a. MULTIPLE PITSTOPS IN A SINGLE RACE ─────────────────────────
    if any(w in q for w in ['pitted more than','pit more than','pitstops in a single','pit stops in a single',
                             'pits in a single','pitted','multiple pit']):
        m_n=re.search(r'(?:more than|over|exceed)\s+(\d+)',q)
        threshold=int(m_n.group(1)) if m_n else 3
        counts=ps.groupby(['Year','Grand Prix','Driver']).size().reset_index(name='Stops')
        over=counts[counts['Stops']>threshold]
        if len(over):
            top=over.sort_values('Stops',ascending=False).iloc[0]
            total=len(over)
            return "Yes — {} instances of drivers making more than {} pitstops in a single race. The most was {} stops by {} at the {} {} Grand Prix.".format(
                total,threshold,int(top['Stops']),top['Driver'],int(top['Year']),top['Grand Prix'])
        return "No driver has made more than {} pitstops in a single race in this dataset (data from 1994).".format(threshold)

    # ── 30. PITSTOPS ──────────────────────────────────────────────────────
    if any(w in q for w in ['pitstop','pit stop','pit-stop','pitstops','pit stops']):
        if any(w in q for w in ['slowest','longest','worst']):
            clean=ps.assign(T=pd.to_numeric(ps['Time'],errors='coerce')).pipe(lambda d: d[d['T']>1])
            r=clean.nlargest(1,'T').iloc[0]
            return "The slowest recorded pitstop was {}s by {} ({}) at the {} {} Grand Prix.".format(r['Time'],r['Driver'],r['Car'],r['Year'],r['Grand Prix'])
        if any(w in q for w in ['fastest','quickest','shortest']):
            clean=ps.assign(T=pd.to_numeric(ps['Time'],errors='coerce')).pipe(lambda d: d[d['T']>1])
            r=clean.loc[clean['T'].idxmin()]
            return "The fastest pitstop on record was {}s by {} ({}) at the {} {} Grand Prix.".format(r['Time'],r['Driver'],r['Car'],r['Year'],r['Grand Prix'])
        if 'team' in q or (team and 'most' in q):
            top=ps.groupby('Car').size().sort_values(ascending=False)
            return "{} has made the most pitstops as a team with {} total.".format(top.index[0], top.iloc[0])
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(ps,'Driver',d)
            return "{} has made {} pitstops in total (data from 1994).".format(d, len(rows))
        top=ps.groupby('Driver').size().sort_values(ascending=False)
        return "{} has made the most pitstops with {} total.".format(top.index[0], top.iloc[0])

    # ── 30b. MOST POLES/PODIUMS/RACES WITHOUT A WIN ──────────────────────
    # e.g. "most poles without a win", "most podiums without winning"
    if any(w in q for w in ['without a win','without winning','without ever winning','no wins',
                             'without a race win','without a victory','winless']):
        winners=set(rs['Winner'].unique())
        if any(w in q for w in ['pole','poles']):
            poles_by_driver=sg[sg['Pos']==1].groupby('Driver').size()
            no_win_poles=poles_by_driver[~poles_by_driver.index.isin(winners)].sort_values(ascending=False)
            if len(no_win_poles):
                lines="\n".join("  {}: {} pole{}".format(d,c,'s' if c!=1 else '') for d,c in no_win_poles.head(5).items())
                return "Most pole positions without ever winning a race:\n{}".format(lines)
        if any(w in q for w in ['podium','podiums']):
            pods=rd[pd.to_numeric(rd['Pos'],errors='coerce')<=3].groupby('Driver').size()
            no_win_pods=pods[~pods.index.isin(winners)].sort_values(ascending=False)
            if len(no_win_pods):
                lines="\n".join("  {}: {} podium{}".format(d,c,'s' if c!=1 else '') for d,c in no_win_pods.head(5).items())
                return "Most podiums without ever winning a race:\n{}".format(lines)
        if any(w in q for w in ['race','races','start','starts','most']):
            race_counts=rd.groupby('Driver').size()
            no_win_races=race_counts[~race_counts.index.isin(winners)].sort_values(ascending=False)
            if len(no_win_races):
                lines="\n".join("  {}: {} race{}".format(d,c,'s' if c!=1 else '') for d,c in no_win_races.head(5).items())
                return "Most race starts without ever winning a race:\n{}".format(lines)

    # ── 31. POLE POSITIONS ────────────────────────────────────────────────
    if any(w in q for w in ['pole position','pole positions','poles','on pole']):
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(sg,'Driver',d); rows=rows[rows['Pos']==1]
            return "{} has taken {} pole positions.".format(d, len(rows))
        if year and gp:
            rows=sg[(sg['Year']==year)&sg['Grand Prix'].str.contains(gp,case=False,na=False)&(sg['Pos']==1)]
            if len(rows): return "{} was on pole for the {} {} Grand Prix.".format(rows.iloc[0]['Driver'],year,gp)
        if year:
            top=sg[(sg['Year']==year)&(sg['Pos']==1)].groupby('Driver').size().sort_values(ascending=False)
            return "{} had the most poles in {} with {}.".format(top.index[0], year, top.iloc[0])
        top=sg[sg['Pos']==1].groupby('Driver').size().sort_values(ascending=False)
        m_n=re.search(r'top\s*(\d+)',q)
        if m_n or 'top' in q or 'list' in q or 'rank' in q:
            n=int(m_n.group(1)) if m_n else 10
            return "Top {} drivers by pole positions:\n".format(n)+'\n'.join("  {}. {}: {}".format(i+1,d,c) for i,(d,c) in enumerate(top.head(n).items()))
        return "{} has the most pole positions of all time with {}.".format(top.index[0], top.iloc[0])

    # ── 32. PODIUMS ───────────────────────────────────────────────────────
    if any(w in q for w in ['podium','podiums','podium finishes']):
        # Two-driver shared podium check
        if len(drivers)==2 and any(w in q for w in ['shared','together','same','both']):
            d1,d2=drivers[0],drivers[1]
            last1,last2=d1.split()[-1],d2.split()[-1]
            p1=rd[(rd['Driver'].str.contains(r'\b'+re.escape(last1)+r'\b',case=False,na=False,regex=True))&(pd.to_numeric(rd['Pos'],errors='coerce')<=3)].set_index(['Year','Grand Prix'])
            p2=rd[(rd['Driver'].str.contains(r'\b'+re.escape(last2)+r'\b',case=False,na=False,regex=True))&(pd.to_numeric(rd['Pos'],errors='coerce')<=3)].set_index(['Year','Grand Prix'])
            shared=p1.index.intersection(p2.index)
            return "{} and {} shared the podium {} time{}.".format(d1,d2,len(shared),'s' if len(shared)!=1 else '')
        # Driver podiums at specific GP
        if gp and len(drivers)==1:
            d=drivers[0]; rows=dmatch(rd,'Driver',d)
            rows_gp=rows[rows['Grand Prix'].str.contains(gp,case=False,na=False)]
            pods=rows_gp[pd.to_numeric(rows_gp['Pos'],errors='coerce')<=3]
            if len(pods):
                details=', '.join("{} (P{})".format(r['Year'],r['Pos']) for _,r in pods.sort_values('Year').iterrows())
                return "{} has {} podium{} at the {} Grand Prix: {}.".format(d,len(pods),'s' if len(pods)!=1 else '',gp,details)
            return "{} has never finished on the podium at the {} Grand Prix in this dataset.".format(d,gp)
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(rd,'Driver',d)
            n=len(rows[pd.to_numeric(rows['Pos'],errors='coerce')<=3])
            return "{} has finished on the podium {} times.".format(d, n)
        top=rd[pd.to_numeric(rd['Pos'],errors='coerce')<=3].groupby('Driver').size().sort_values(ascending=False)
        m_n=re.search(r'top\s*(\d+)',q)
        if m_n or 'top' in q or 'list' in q or 'rank' in q:
            n=int(m_n.group(1)) if m_n else 10
            return "Top {} drivers by podium finishes:\n".format(n)+'\n'.join("  {}. {}: {}".format(i+1,d,c) for i,(d,c) in enumerate(top.head(n).items()))
        return "{} has the most podium finishes of all time with {}.".format(top.index[0], top.iloc[0])

    # ── 32a. FINISHED OUTSIDE POINTS ─────────────────────────────────────
    # e.g. "how many races did hamilton finish outside points"
    if any(w in q for w in ['outside points','outside the points','no points','zero points',
                             'pointless','score no points','scored no points','scored 0']):
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(rd,'Driver',d)
            rows=rows[pd.to_numeric(rows['PTS'],errors='coerce')==0]
            if yr_start and yr_end:
                rows=rows[(rows['Year']>=yr_start)&(rows['Year']<=yr_end)]
                return "{} finished outside the points {} times from {} to {}.".format(d,len(rows),yr_start,yr_end)
            if year:
                rows=rows[rows['Year']==year]
                return "{} finished outside the points {} times in {}.".format(d,len(rows),year)
            return "{} finished outside the points {} times in their career.".format(d,len(rows))
        top=rd[pd.to_numeric(rd['PTS'],errors='coerce')==0].groupby('Driver').size().sort_values(ascending=False)
        return "{} has the most pointless finishes with {}.".format(top.index[0],top.iloc[0])

    # ── 32b. LEAST POINTS WITH RACE THRESHOLD ────────────────────────────
    # e.g. "who has the least points scored having raced for more than 100 races"
    if any(w in q for w in ['least points','least amount','fewest points','lowest points']) and any(w in q for w in ['more than','over','at least','raced']):
        m_n=re.search(r'(?:more than|over|at least|than)\s+(\d+)',q)
        threshold=int(m_n.group(1)) if m_n else 100
        race_counts=rd.groupby('Driver').size().reset_index(name='Races')
        eligible=race_counts[race_counts['Races']>threshold]
        pts=dd.groupby('Driver')['PTS'].sum().reset_index(name='TotalPTS')
        merged=eligible.merge(pts,on='Driver',how='left').fillna(0).sort_values('TotalPTS')
        if len(merged):
            lines="\n".join("  {}: {:.0f} pts ({} races)".format(r['Driver'],r['TotalPTS'],r['Races']) for _,r in merged.head(5).iterrows())
            return "Drivers with the least championship points (more than {} races):\n{}".format(threshold,lines)

    # ── 33. POINTS ────────────────────────────────────────────────────────
    if any(w in q for w in ['points','scored']):
        if yr_start and yr_end and len(drivers)==1:
            d=drivers[0]; rows=dmatch(ds,'Driver',d)
            rows=rows[(rows['Year']>=yr_start)&(rows['Year']<=yr_end)]
            if len(rows):
                total=rows['PTS'].sum()
                lines="\n".join("  {}: {:.0f} pts (P{})".format(r['Year'],r['PTS'],r['Pos']) for _,r in rows.sort_values('Year').iterrows())
                return "{} scored {:.0f} total points from {} to {}:\n{}".format(d,total,yr_start,yr_end,lines)
        if year and len(drivers)==1:
            d=drivers[0]; rows=dmatch(ds,'Driver',d); rows=rows[rows['Year']==year]
            if len(rows): return "{} scored {:.0f} points in {} (P{}).".format(d, rows.iloc[0]['PTS'], year, rows.iloc[0]['Pos'])
        if year:
            top=ds[ds['Year']==year].sort_values('PTS',ascending=False).head(5)
            lines="\n".join("  {}. {}: {:.0f} pts".format(i+1,r['Driver'],r['PTS']) for i,(_,r) in enumerate(top.iterrows()))
            return "Top points scorers in {}:\n{}".format(year, lines)
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(dd,'Driver',d)
            return "{} has scored {:.0f} total championship points.".format(d, rows['PTS'].sum())

    # ── 34. HOW MANY RACES (season / all time) ────────────────────────────
    if any(w in q for w in ['how many races','number of races','races were held','races held','races total','total races','f1 races']) and not any(w in q for w in ['competed','started','won','win','hosted','circuit','dnf','retired','retirement']):
        if year:
            return "There were {} races in the {} Formula 1 season.".format(rs[rs['Year']==year].shape[0], year)
        return "There have been {} Formula 1 races in total from 1950 to 2022.".format(rs.shape[0])

    # ── 35. DIFFERENT WINNERS ─────────────────────────────────────────────
    if 'different' in q and any(w in q for w in ['winner','winners']):
        if year:
            return "There were {} different race winners in the {} season.".format(rs[rs['Year']==year]['Winner'].nunique(), year)
    if any(w in q for w in ['most different winners','year with most winners','most winners in a season']):
        top=rs.groupby('Year')['Winner'].nunique().sort_values(ascending=False)
        return "{} had the most different race winners with {}.".format(top.index[0], top.iloc[0])

    # ── 36. RETIREMENTS / DNF ─────────────────────────────────────────────
    if any(w in q for w in ['dnf','retired from','not classified','did not finish','retirement','retirements']):
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(rd,'Driver',d); rows=rows[rows['Pos']=='NC']
            if year:
                rows=rows[rows['Year']==year]
                return "{} had {} DNF{} in {}.".format(d, len(rows), 's' if len(rows)!=1 else '', year)
            return "{} has retired from {} races in total.".format(d, len(rows))
        if year:
            top=rd[(rd['Year']==year)&(rd['Pos']=='NC')].groupby('Driver').size().sort_values(ascending=False)
            if len(top): return "Most DNFs in {}: {} with {}.".format(year, top.index[0], top.iloc[0])
        top=rd[rd['Pos']=='NC'].groupby('Driver').size().sort_values(ascending=False)
        return "{} has the most retirements with {}.".format(top.index[0], top.iloc[0])

    # ── 36a. QUALIFYING EXITS (Q1/Q2/Q3) ────────────────────────────────
    # e.g. "who had the most Q3 exits", "most Q1 eliminations"
    if any(w in q for w in ['q1','q2','q3']) and any(w in q for w in ['exit','exits','elimination','eliminations',
                                                                       'appearance','appearances','knocked out',
                                                                       'reached','made it','got into','most']):
        # Only works for 2006+ when Q1/Q2/Q3 format was introduced
        modern=qu[qu['Q1'].notna()&(qu['Q1'].astype(str).str.strip()!='')]
        if 'q3' in q:
            if any(w in q for w in ['appearance','appearances','reached','made it','got into']):
                q3_drivers=modern[modern['Q3'].notna()&(modern['Q3'].astype(str).str.strip()!='')].groupby('Driver').size().sort_values(ascending=False)
                if len(drivers)==1:
                    d=drivers[0]; rows=dmatch(modern[modern['Q3'].notna()&(modern['Q3'].astype(str).str.strip()!='')],'Driver',d)
                    return "{} reached Q3 {} times.".format(d,len(rows))
                lines="\n".join("  {}: {} times".format(d,c) for d,c in q3_drivers.head(10).items())
                return "Most Q3 appearances (2006+):\n{}".format(lines)
            # Q3 exits = made it to Q3 (set a Q3 time, positions 1-10)
            q3_drivers=modern[modern['Q3'].notna()&(modern['Q3'].astype(str).str.strip()!='')].groupby('Driver').size().sort_values(ascending=False)
            if len(drivers)==1:
                d=drivers[0]; rows=dmatch(modern[modern['Q3'].notna()&(modern['Q3'].astype(str).str.strip()!='')],'Driver',d)
                return "{} made it to Q3 {} times.".format(d,len(rows))
            lines="\n".join("  {}: {} times".format(d,c) for d,c in q3_drivers.head(10).items())
            return "Most Q3 appearances (2006+):\n{}".format(lines)
        if 'q2' in q:
            # Q2 exit = had Q2 time but no Q3 time (eliminated in Q2)
            q2_exit=modern[(modern['Q2'].notna()&(modern['Q2'].astype(str).str.strip()!=''))&(modern['Q3'].isna()|(modern['Q3'].astype(str).str.strip()==''))]
            if len(drivers)==1:
                d=drivers[0]; rows=dmatch(q2_exit,'Driver',d)
                return "{} was eliminated in Q2 {} times.".format(d,len(rows))
            top=q2_exit.groupby('Driver').size().sort_values(ascending=False)
            lines="\n".join("  {}: {} times".format(d,c) for d,c in top.head(10).items())
            return "Most Q2 eliminations (2006+):\n{}".format(lines)
        if 'q1' in q:
            # Q1 exit = had Q1 time but no Q2 time (eliminated in Q1)
            q1_exit=modern[(modern['Q1'].notna()&(modern['Q1'].astype(str).str.strip()!=''))&(modern['Q2'].isna()|(modern['Q2'].astype(str).str.strip()==''))]
            if len(drivers)==1:
                d=drivers[0]; rows=dmatch(q1_exit,'Driver',d)
                return "{} was eliminated in Q1 {} times.".format(d,len(rows))
            top=q1_exit.groupby('Driver').size().sort_values(ascending=False)
            lines="\n".join("  {}: {} times".format(d,c) for d,c in top.head(10).items())
            return "Most Q1 eliminations (2006+):\n{}".format(lines)

    # ── 37. QUALIFYING ────────────────────────────────────────────────────
    if any(w in q for w in ['qualifying','qualify','qualification','quali']):
        if year and gp:
            rows=qu[(qu['Year']==year)&qu['Grand Prix'].str.contains(gp,case=False,na=False)].sort_values('Pos')
            if len(rows): return "Qualifying for {} {} Grand Prix:\n".format(year,gp)+rows.head(5)[['Pos','Driver','Car','Time']].to_string(index=False)

    # ── 38. CHAMPIONSHIP STANDINGS FOR A YEAR ────────────────────────────
    if any(w in q for w in ['standings','championship table','final standings','driver standings','championship standings']) and year:
        if any(w in q for w in ['constructor','team','constructors']):
            top=cs[cs['Year']==year].sort_values('PTS',ascending=False).head(10)
            lines="\n".join("  P{}  {}  {:.0f} pts".format(r['Pos'],r['Team'],r['PTS']) for _,r in top.iterrows())
            return "{} Constructors Championship standings:\n{}".format(year, lines)
        top=ds[ds['Year']==year].sort_values('PTS',ascending=False).head(10)
        lines="\n".join("  P{}  {}  {:.0f} pts".format(r['Pos'],r['Driver'],r['PTS']) for _,r in top.iterrows())
        return "{} Drivers Championship standings:\n{}".format(year, lines)

    # ── 39. HOW MANY RACES HAS DRIVER COMPETED IN ─────────────────────────
    if any(w in q for w in ['competed','appearances','how many races has','how many races did']) and len(drivers)==1:
        if 'won' not in q and 'win' not in q:
            d=drivers[0]; rows=dmatch(rd,'Driver',d)
            return "{} has competed in {} Formula 1 races.".format(d, len(rows))

    # ── 40. MOST RACES PARTICIPATED ───────────────────────────────────────
    if any(w in q for w in ['most starts','most appearances','competed in most']) or ('most races' in q and 'won' not in q and not year and not team and not drivers):
        top=rd.groupby('Driver').size().sort_values(ascending=False)
        return "{} has competed in the most F1 races with {} starts.".format(top.index[0], top.iloc[0])

    # ── 41. CIRCUIT HOSTED MOST RACES ─────────────────────────────────────
    if any(w in q for w in ['most hosted','most times hosted','hosted the most','circuit has hosted most']):
        top=rs.groupby('Grand Prix').size().sort_values(ascending=False)
        return "{} has hosted the most F1 Grands Prix with {} editions.".format(top.index[0], top.iloc[0])

    # ── 42. LAST RACE OF A SEASON ─────────────────────────────────────────
    if any(w in q for w in ['last race','final race','season finale','last grand prix','final grand prix']) and year:
        r=rs[rs['Year']==year].iloc[-1]
        return "The last race of {} was the {} Grand Prix, won by {} ({}).".format(year, r['Grand Prix'], r['Winner'], r['Car'])

    # ── 43. RACE FULL RESULTS ─────────────────────────────────────────────
    if any(w in q for w in ['race results','full results','finishing order','race order','top 10']) and year and gp:
        rows=rd[(rd['Year']==year)&rd['Grand Prix'].str.contains(gp,case=False,na=False)]
        rows_s=rows.assign(PN=pd.to_numeric(rows['Pos'],errors='coerce')).sort_values('PN')
        if len(rows_s): return "{} {} Grand Prix results:\n".format(year,gp)+rows_s.head(10)[['Pos','Driver','Car','PTS']].to_string(index=False)

    # ── 44. DATE / WHEN ───────────────────────────────────────────────────
    if any(w in q for w in ['when','what date','what day','took place','take place']):
        if year and gp:
            rows=rs[(rs['Year']==year)&rs['Grand Prix'].str.contains(gp,case=False,na=False)]
            if len(rows): return "The {} {} Grand Prix took place on {}. Won by {} ({}).".format(year,gp,rows.iloc[0]['Date'],rows.iloc[0]['Winner'],rows.iloc[0]['Car'])
        if gp and not year:
            asc='first' in q or 'oldest' in q or 'earliest' in q
            rows=rs[rs['Grand Prix'].str.contains(gp,case=False,na=False)].sort_values('Year',ascending=asc)
            if len(rows):
                if asc: r=rows.iloc[0]; return "The first {} Grand Prix took place on {}, in {}.".format(gp, r['Date'], r['Year'])
                lines="\n".join("  {}: {}".format(r['Year'],r['Date']) for _,r in rows.head(10).iterrows())
                return "Recent {} Grand Prix dates:\n{}".format(gp, lines)

    # ── 45. HOW MANY LAPS ────────────────────────────────────────────────
    if any(w in q for w in ['how many laps','laps was','laps in','laps did','laps were']) and year and gp:
        rows=rs[(rs['Year']==year)&rs['Grand Prix'].str.contains(gp,case=False,na=False)]
        if len(rows): return "The {} {} Grand Prix was run over {} laps.".format(year, gp, int(rows.iloc[0]['Laps']))

    # ── 46. RACE DURATION ────────────────────────────────────────────────
    if any(w in q for w in ['how long','race time','duration','winning time']) and year and gp:
        rows=rs[(rs['Year']==year)&rs['Grand Prix'].str.contains(gp,case=False,na=False)]
        if len(rows): return "The {} {} Grand Prix winning time was {} ({}).".format(year, gp, rows.iloc[0]['Time'], rows.iloc[0]['Winner'])

    # ── 47. WHICH TEAM DOES DRIVER DRIVE FOR ─────────────────────────────
    if any(w in q for w in ['which team','what team','who does','drive for','drives for','racing for','race for','what car']) and len(drivers)==1:
        d=drivers[0]; rows=dmatch(ds,'Driver',d)
        if year:
            yr_rows=rows[rows['Year']==year]
            if len(yr_rows): return "{} drove for {} in the {} season.".format(d, yr_rows.iloc[0]['Car'], year)
        rows2=rows[['Year','Car']].drop_duplicates().sort_values('Year',ascending=False)
        if len(rows2):
            lines="\n".join("  {}: {}".format(r['Year'],r['Car']) for _,r in rows2.head(10).iterrows())
            return "{} drove for these teams:\n{}".format(d, lines)

    # ── 48. WHEN DID DRIVER JOIN TEAM ────────────────────────────────────
    if any(w in q for w in ['start driving','started driving','join','joined','first drive','first drove',
                             'begin driving','move to','moved to','sign for','signed for']) and len(drivers)==1 and team:
        d=drivers[0]; rows=dmatch(ds,'Driver',d)
        team_rows=rows[rows['Car'].str.lower().str.startswith(team.lower())]
        if not len(team_rows):
            team_rows=rows[rows['Car'].str.contains(team,case=False,na=False)]
        if len(team_rows):
            fy=team_rows['Year'].min(); ly=team_rows['Year'].max(); cn=team_rows.sort_values('Year').iloc[0]['Car']
            if fy==ly: return "{} drove for {} in {}.".format(d, cn, fy)
            return "{} first drove for {} in {} and was with them through {} ({} seasons).".format(d, cn, fy, ly, ly-fy+1)
        return "No records found for {} driving for {} in this dataset (1950-2022).".format(d, team)

    # ── 49. CAREER STATS SUMMARY ─────────────────────────────────────────
    if any(w in q for w in ['career','stats','statistics','profile']) and len(drivers)==1:
        d=drivers[0]
        wins=len(dmatch(rs,'Winner',d))
        rows_sg3=dmatch(sg,'Driver',d); poles=len(rows_sg3[rows_sg3['Pos']==1])
        rows_rd4=dmatch(rd,'Driver',d); podiums=len(rows_rd4[pd.to_numeric(rows_rd4['Pos'],errors='coerce')<=3])
        races=len(rows_rd4)
        rows_ds4=dmatch(ds,'Driver',d); champs=len(rows_ds4[rows_ds4['Pos']=='1'])
        fl_n=len(dmatch(fl,'Driver',d))
        return ("{} career stats:\n  Races:          {}\n  Wins:           {}\n  Podiums:        {}\n"
                "  Pole positions: {}\n  Fastest laps:   {}\n  Championships:  {}").format(d,races,wins,podiums,poles,fl_n,champs)

    # ── 50. YOUNGEST WORLD CHAMPION ──────────────────────────────────────
    if any(w in q for w in ['youngest champion','youngest world champion','youngest driver to win','youngest title']):
        return ("Sebastian Vettel became the youngest F1 World Champion in 2010 at age 23. "
                "(Note: this dataset does not include driver birthdays, this is from general F1 knowledge.)")

    # ── 52. LIST DRIVERS BY SURNAME ──────────────────────────────────────
    if any(w in q for w in ['list all','list drivers','all drivers','drivers named','drivers with','drivers with last name','find driver','search driver']) and len(drivers)==1:
        d=drivers[0]; last=d.split()[-1]
        found=ds[ds['Driver'].str.contains(r'\b'+re.escape(last)+r'\b', case=False, na=False, regex=True)]['Driver'].unique()
        if len(found):
            return "Drivers found with '{}' in their name:\n{}".format(last, '\n'.join('  '+n for n in sorted(found)))
        return "No drivers found with '{}' in their name in this dataset.".format(last)

    # ── 53. DRIVER EXISTENCE CHECK ────────────────────────────────────────
    if any(w in q for w in ['was there ever','is there a driver','was there a driver','has there been','has a driver','exist','was he a driver','was she a driver']) and len(drivers)==1:
        d=drivers[0]; last=d.split()[-1]
        rows=dmatch(ds,'Driver',d)
        if len(rows):
            first_yr=rows['Year'].min(); last_yr=rows['Year'].max()
            seasons=rows['Year'].nunique()
            car=rows.sort_values('Year').iloc[0]['Car']
            return "Yes, {} competed in F1 from {} to {} ({} season{}) starting with {}.".format(
                rows.iloc[0]['Driver'], first_yr, last_yr, seasons, 's' if seasons!=1 else '', car)
        return "No driver named '{}' was found in this dataset (1950-2022).".format(d)

    # ── 54. MULTIPLE WINNERS AT SPECIFIC GP ───────────────────────────────
    if gp and any(w in q for w in ['more than once','multiple times','won it more','won multiple','multiple winners','won it twice']):
        multi=rs[rs['Grand Prix'].str.contains(gp, case=False, na=False)].groupby('Winner').size()
        multi=multi[multi>1].sort_values(ascending=False)
        if len(multi):
            lines="\n".join("  {}: {} times".format(n,c) for n,c in multi.items())
            return "{} drivers have won the {} Grand Prix more than once:\n{}".format(len(multi), gp, lines)
        return "No driver has won the {} Grand Prix more than once in this dataset.".format(gp)

    # ── 55. SLOWEST PITSTOP ───────────────────────────────────────────────
    if any(w in q for w in ['slowest pitstop','slowest pit stop','longest pitstop','longest pit stop','worst pitstop','slowest stop']):
        clean=ps.assign(T=pd.to_numeric(ps['Time'],errors='coerce')).pipe(lambda d: d[d['T']>1])
        r=clean.nlargest(1,'T').iloc[0]
        return "The slowest recorded pitstop was {}s by {} ({}) at the {} {} Grand Prix.".format(r['Time'],r['Driver'],r['Car'],r['Year'],r['Grand Prix'])

    # ── 56. DRIVER PODIUMS AT SPECIFIC GP ────────────────────────────────
    if gp and len(drivers)==1 and any(w in q for w in ['podium','podiums','podium finishes','times on the podium']):
        d=drivers[0]; rows=dmatch(rd,'Driver',d)
        rows_gp=rows[rows['Grand Prix'].str.contains(gp,case=False,na=False)]
        podiums=rows_gp[pd.to_numeric(rows_gp['Pos'],errors='coerce')<=3]
        if len(podiums):
            details=', '.join("{} (P{})".format(r['Year'],r['Pos']) for _,r in podiums.sort_values('Year').iterrows())
            return "{} has finished on the podium at {} {} time{} at the {} Grand Prix ({}).".format(
                d, gp, len(podiums), 's' if len(podiums)!=1 else '', gp, details)
        return "{} has never finished on the podium at the {} Grand Prix in this dataset.".format(d, gp)

    # ── 57. CHAMPIONSHIP HEAD-TO-HEAD ────────────────────────────────────
    if len(drivers)==2 and any(w in q for w in ['ahead','beat','beaten','finished ahead','ended ahead','finished higher','higher in the championship','more championships than']):
        d1,d2=drivers[0],drivers[1]
        sub=ds[ds['Driver'].isin([d1,d2])]
        piv=sub.pivot_table(index='Year',columns='Driver',values='PTS').dropna()
        if d1 in piv.columns and d2 in piv.columns:
            d1_ahead=int((piv[d1]>piv[d2]).sum()); d2_ahead=int((piv[d2]>piv[d1]).sum())
            n=len(piv)
            return ("In {} seasons where both {} and {} competed:\n"
                    "  {} finished higher: {} times\n"
                    "  {} finished higher: {} times").format(n,d1,d2,d1,d1_ahead,d2,d2_ahead)
        return "Not enough shared seasons found for {} and {}.".format(d1,d2)

    # ── 58. SHARED PODIUM COUNT ───────────────────────────────────────────
    if len(drivers)==2 and any(w in q for w in ['shared podium','shared a podium','both on the podium','podium together','same podium']):
        d1,d2=drivers[0],drivers[1]
        last1,last2=d1.split()[-1],d2.split()[-1]
        p1=rd[(rd['Driver'].str.contains(r'\b'+re.escape(last1)+r'\b',case=False,na=False,regex=True)) & (pd.to_numeric(rd['Pos'],errors='coerce')<=3)].set_index(['Year','Grand Prix'])
        p2=rd[(rd['Driver'].str.contains(r'\b'+re.escape(last2)+r'\b',case=False,na=False,regex=True)) & (pd.to_numeric(rd['Pos'],errors='coerce')<=3)].set_index(['Year','Grand Prix'])
        shared=p1.index.intersection(p2.index)
        return "{} and {} shared the podium {} time{} in this dataset.".format(d1,d2,len(shared),'s' if len(shared)!=1 else '')

    # ── 59a. LATEST / FIRST WIN FOR A DRIVER ────────────────────────────
    if len(drivers)==1 and not year and not gp and (
        any(w in q for w in ['last win','latest win','most recent win','last victory','most recent victory',
                              'first win','first victory','last race won','latest victory']) or
        (any(w in q for w in ['last','latest','most recent','first','earliest']) and any(w in q for w in ['win','won','victory']))):
        d=drivers[0]; sub=dmatch(rs,'Winner',d)
        if not len(sub):
            return "No wins found for {} in this dataset.".format(d)
        asc='first' in q or 'earliest' in q
        r=sub.sort_values('Year',ascending=asc).iloc[0]
        label='first' if asc else 'most recent'
        return "The {} win for {} was the {} {} Grand Prix, driving for {}.".format(label,d,int(r['Year']),r['Grand Prix'],r['Car'])

    # ── 59. LATEST / FIRST WIN FOR A TEAM ────────────────────────────────
    if team and any(w in q for w in ['latest win','most recent win','last win','last victory','most recent victory','when did','when was','first win','first victory','earliest win','when was the']):
        sub=rs[rs['Car'].str.contains(team,case=False,na=False)]
        if not len(sub):
            return "No wins found for {} in this dataset.".format(team)
        asc='first' in q or 'earliest' in q
        r=sub.sort_values('Year',ascending=asc).iloc[0]
        label='first' if asc else 'most recent'
        return "The {} win for {} was the {} {} Grand Prix, driven by {}.".format(label,team,r['Year'],r['Grand Prix'],r['Winner'])

    # ── 51. SEASON RESULTS FOR A DRIVER ──────────────────────────────────
    if year and len(drivers)==1 and any(w in q for w in ['results','season','how did','finished','performed']):
        d=drivers[0]; rows=dmatch(rd,'Driver',d); rows=rows[rows['Year']==year][['Grand Prix','Pos','PTS']]
        if len(rows): return "{}'s {} season results:\n{}".format(d, year, rows.to_string(index=False))

    # ── 60. TEAM / DRIVER WITH EXACTLY N WINS ─────────────────────────────
    # e.g. "which team has only one win", "which driver has exactly 5 wins"
    if any(w in q for w in ['only one win','exactly one win','just one win','single win',
                             'only 1 win','exactly 1 win','only two win','exactly two win',
                             'only 2 win','exactly 2 win','only three win','exactly three win',
                             'only 3 win','exactly 3 win','only one race','exactly one race',
                             'just one race','single race win','won only once','won exactly once',
                             'won just once','won only 1','won exactly 1']):
        word_to_num={'one':1,'1':1,'two':2,'2':2,'three':3,'3':3,'four':4,'4':4,'five':5,'5':5}
        target=1
        for word,num in word_to_num.items():
            if word in q: target=num; break
        if any(w in q for w in ['team','constructor','car']):
            counts=rs.groupby('Car').size()
            exact=counts[counts==target].sort_index()
            if len(exact):
                lines="\n".join("  {}".format(t) for t in exact.index[:15])
                return "Teams with exactly {} race win{}:\n{}".format(target,'s' if target!=1 else '',lines)
            return "No team has exactly {} race win{} in this dataset.".format(target,'s' if target!=1 else '')
        counts=rs.groupby('Winner').size()
        exact=counts[counts==target].sort_index()
        if len(exact):
            lines="\n".join("  {}".format(d) for d in exact.index[:15])
            return "Drivers with exactly {} race win{}:\n{}".format(target,'s' if target!=1 else '',lines)
        return "No driver has exactly {} race win{} in this dataset.".format(target,'s' if target!=1 else '')

    # ── 63. WON ON DEBUT / FIRST RACE ─────────────────────────────────────
    # e.g. "who won on their debut", "who won their first race"
    if any(w in q for w in ['won on debut','won their debut','won his debut','won on their first',
                             'won on their debut','won their first race','won his first race',
                             'won first race','win on debut','debut win','won in their first']):
        first_races=rd.groupby('Driver').apply(lambda x: x.sort_values('Year').iloc[0],include_groups=False).reset_index()
        debut_winners=first_races[first_races['Pos']=='1']
        if len(debut_winners):
            lines="\n".join("  {} — {} {} Grand Prix".format(r['Driver'],r['Year'],r['Grand Prix']) for _,r in debut_winners.sort_values('Year').iterrows())
            return "Drivers who won on their F1 debut:\n{}".format(lines)
        return "No driver in this dataset won on their F1 debut."

    # ── 64. WON FOR MULTIPLE TEAMS ─────────────────────────────────────────
    # e.g. "which drivers have won for multiple teams", "how many teams has hamilton won for"
    if any(w in q for w in ['multiple teams','different teams','how many teams','won for more than one',
                             'won with different','won for different']):
        if len(drivers)==1:
            d=drivers[0]; rows=dmatch(rs,'Winner',d)
            teams=rows['Car'].unique()
            return "{} has won races for {} different team{}: {}.".format(d,len(teams),'s' if len(teams)!=1 else '',', '.join(sorted(teams)))
        driver_teams=rs.groupby('Winner')['Car'].nunique()
        multi=driver_teams[driver_teams>1].sort_values(ascending=False)
        if len(multi):
            lines="\n".join("  {}: {} teams".format(d,c) for d,c in multi.head(10).items())
            return "Drivers who have won races for multiple teams:\n{}".format(lines)

    # ── 68. DRIVERS WHO HAVE WON IN SPECIFIC YEAR/SEASON ─────────────────
    # e.g. "which drivers won a race in 2020", "all winners in 2019"
    if year and any(w in q for w in ['which driver','all winners','every winner','all the winners','list winners','who all won','list of winners']):
        winners=rs[rs['Year']==year].groupby('Winner').size().sort_values(ascending=False)
        if len(winners):
            lines="\n".join("  {}: {} win{}".format(d,c,'s' if c!=1 else '') for d,c in winners.items())
            return "Race winners in {}:\n{}".format(year,lines)

    return None


# ─────────────────────────────────────────────
# 4. FORMAT ANSWER (LLM only used here)
# ─────────────────────────────────────────────

REFUSAL_PHRASES = [
    "i don't have","i'm sorry","as an ai","i cannot","i'm not able",
    "developed by","feel free","i am an ai","i apologize",
    "not designed","cannot provide","as a language model","i don't know",
]

def format_answer(question: str, raw_result: str) -> str:
    """Use LLM to turn raw data result into a natural sentence."""
    try:
        prompt = (
            "You are an F1 data reporter. Convert the data result into one clear sentence.\n"
            "Output ONLY the sentence. No disclaimers. No extra facts. Report only what the data says.\n\n"
            f"Question: {question}\n"
            f"Data: {raw_result}\n\n"
            "Sentence:"
        )
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 80}
        }, timeout=30)
        response.raise_for_status()
        answer = response.json()["response"].strip()

        if any(p in answer.lower() for p in REFUSAL_PHRASES):
            return raw_result  # fall back to raw data
        return answer
    except Exception:
        return raw_result  # if Ollama fails, just return the raw data


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def ask(question: str, dfs: dict, verbose: bool = False) -> str:
    raw = query_data(question, dfs)

    if raw is None:
        return "❓ I don't have a pattern for that question yet. Try rephrasing, e.g. 'how many wins does X have' or 'who won the Y grand prix'."

    if verbose:
        print(f"\n  Raw result: {raw[:200]}")

    # If result is already a well-formed sentence, skip LLM formatting
    if raw.endswith('.') and '\n' not in raw:
        return raw

    # Multi-line results (tables, lists) — return directly without LLM
    if '\n' in raw:
        return raw

    # Single value — ask LLM to make it sound natural
    return format_answer(question, raw)


# ─────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        dfs = load_data()
        print(f"Q: {question}")
        print(f"A: {ask(question, dfs, verbose=True)}")
        return

    dfs = load_data()

    print("=" * 54)
    print("  F1 Assistant — 1950 to 2022")
    print("  'verbose' = show raw result | 'quit' = exit")
    print("=" * 54 + "\n")

    verbose = False
    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                print("Goodbye!")
                break
            if question.lower() == "verbose":
                verbose = not verbose
                print(f"  Verbose mode: {'ON' if verbose else 'OFF'}\n")
                continue

            print(f"F1 Bot: {ask(question, dfs, verbose=verbose)}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
