% Chess mate-in-one solver
% Board squares
square(1..8, 1..8).

% Directions for pieces
rook_dir(1,0). rook_dir(-1,0). rook_dir(0,1). rook_dir(0,-1).
bishop_dir(1,1). bishop_dir(1,-1). bishop_dir(-1,1). bishop_dir(-1,-1).
queen_dir(1,0). queen_dir(-1,0). queen_dir(0,1). queen_dir(0,-1).
queen_dir(1,1). queen_dir(1,-1). queen_dir(-1,1). queen_dir(-1,-1).
knight_move(1,2). knight_move(2,1). knight_move(-1,2). knight_move(-2,1).
knight_move(1,-2). knight_move(2,-1). knight_move(-1,-2). knight_move(-2,-1).
king_move(1,0). king_move(-1,0). king_move(0,1). king_move(0,-1).
king_move(1,1). king_move(1,-1). king_move(-1,1). king_move(-1,-1).

% Opponent color
opponent(white, black). opponent(black, white).

% All pieces on board
occupied(X, Y, Color, Type) :- piece(Color, Type, X, Y).
occupied(X, Y, Color, king) :- king(Color, X, Y).

% Is a square occupied by any piece?
any_piece(X, Y) :- occupied(X, Y, _, _).

% Sliding piece path blocking check
% For a sliding piece moving from (FX,FY) in direction (DX,DY), step N
path_square(FX, FY, DX, DY, 1, FX+DX, FY+DY) :-
    square(FX, FY), rook_dir(DX, DY),
    square(FX+DX, FY+DY).
path_square(FX, FY, DX, DY, 1, FX+DX, FY+DY) :-
    square(FX, FY), bishop_dir(DX, DY),
    square(FX+DX, FY+DY).

% Generic sliding: can reach (TX,TY) from (FX,FY) in direction (DX,DY) with no blockers
% Use step-by-step approach
slide_step(FX, FY, DX, DY, FX+DX, FY+DY) :-
    square(FX, FY), queen_dir(DX, DY),
    square(FX+DX, FY+DY).

slide_reach(FX, FY, DX, DY, TX, TY) :-
    slide_step(FX, FY, DX, DY, TX, TY),
    not any_piece(TX, TY).
slide_reach(FX, FY, DX, DY, TX, TY) :-
    slide_step(FX, FY, DX, DY, TX, TY),
    any_piece(TX, TY).

% Intermediate squares blocked
slide_blocked(FX, FY, DX, DY, TX, TY) :-
    slide_step(FX, FY, DX, DY, MX, MY),
    slide_reach(MX, MY, DX, DY, TX, TY),
    any_piece(MX, MY).

% Can a sliding piece reach TX,TY from FX,FY in direction DX,DY (clear path)?
slide_clear(FX, FY, DX, DY, TX, TY) :-
    slide_step(FX, FY, DX, DY, TX, TY).
slide_clear(FX, FY, DX, DY, TX, TY) :-
    slide_step(FX, FY, DX, DY, MX, MY),
    not any_piece(MX, MY),
    slide_clear(MX, MY, DX, DY, TX, TY).

% Piece attacks (ignoring whose turn it is, for check detection)
% Rook attacks
attacks(Color, rook, FX, FY, TX, TY) :-
    occupied(FX, FY, Color, rook),
    square(TX, TY),
    rook_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    FX != TX.
attacks(Color, rook, FX, FY, TX, TY) :-
    occupied(FX, FY, Color, rook),
    square(TX, TY),
    rook_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    FY != TY.

% Bishop attacks
attacks(Color, bishop, FX, FY, TX, TY) :-
    occupied(FX, FY, Color, bishop),
    square(TX, TY),
    bishop_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    FX != TX.

% Queen attacks
attacks(Color, queen, FX, FY, TX, TY) :-
    occupied(FX, FY, Color, queen),
    square(TX, TY),
    queen_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    FX != TX.
attacks(Color, queen, FX, FY, TX, TY) :-
    occupied(FX, FY, Color, queen),
    square(TX, TY),
    queen_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    FY != TY.

% Knight attacks
attacks(Color, knight, FX, FY, TX, TY) :-
    occupied(FX, FY, Color, knight),
    knight_move(DX, DY),
    TX = FX + DX, TY = FY + DY,
    square(TX, TY).

% King attacks
attacks(Color, king, FX, FY, TX, TY) :-
    king(Color, FX, FY),
    king_move(DX, DY),
    TX = FX + DX, TY = FY + DY,
    square(TX, TY).

% Pawn attacks
attacks(white, pawn, FX, FY, TX, TY) :-
    occupied(FX, FY, white, pawn),
    TY = FY + 1,
    (TX = FX + 1 ; TX = FX - 1),
    square(TX, TY).
attacks(black, pawn, FX, FY, TX, TY) :-
    occupied(FX, FY, black, pawn),
    TY = FY - 1,
    (TX = FX + 1 ; TX = FX - 1),
    square(TX, TY).

% Split pawn attacks to avoid disjunction in body
attacks(white, pawn, FX, FY, FX+1, FY+1) :-
    occupied(FX, FY, white, pawn),
    square(FX+1, FY+1).
attacks(white, pawn, FX, FY, FX-1, FY+1) :-
    occupied(FX, FY, white, pawn),
    square(FX-1, FY+1).
attacks(black, pawn, FX, FY, FX+1, FY-1) :-
    occupied(FX, FY, black, pawn),
    square(FX+1, FY-1).
attacks(black, pawn, FX, FY, FX-1, FY-1) :-
    occupied(FX, FY, black, pawn),
    square(FX-1, FY-1).

% Remove duplicate pawn attacks from disjunction above - just keep split rules
% (The disjunction rules above will cause issues, remove them)

% Legal moves for the side to move
% Pawn moves (white moves up, black moves down)
pawn_push(white, FX, FY, FX, FY+1) :-
    occupied(FX, FY, white, pawn),
    square(FX, FY+1),
    not any_piece(FX, FY+1).
pawn_push(white, FX, FY, FX, FY+2) :-
    occupied(FX, FY, white, pawn),
    FY = 2,
    not any_piece(FX, FY+1),
    not any_piece(FX, FY+2),
    square(FX, FY+2).
pawn_push(black, FX, FY, FX, FY-1) :-
    occupied(FX, FY, black, pawn),
    square(FX, FY-1),
    not any_piece(FX, FY-1).
pawn_push(black, FX, FY, FX, FY-2) :-
    occupied(FX, FY, black, pawn),
    FY = 7,
    not any_piece(FX, FY-1),
    not any_piece(FX, FY-2),
    square(FX, FY-2).

% Pawn captures
pawn_capture(white, FX, FY, FX+1, FY+1) :-
    occupied(FX, FY, white, pawn),
    square(FX+1, FY+1),
    occupied(FX+1, FY+1, black, _).
pawn_capture(white, FX, FY, FX-1, FY+1) :-
    occupied(FX, FY, white, pawn),
    square(FX-1, FY+1),
    occupied(FX-1, FY+1, black, _).
pawn_capture(black, FX, FY, FX+1, FY-1) :-
    occupied(FX, FY, black, pawn),
    square(FX+1, FY-1),
    occupied(FX+1, FY-1, white, _).
pawn_capture(black, FX, FY, FX-1, FY-1) :-
    occupied(FX, FY, black, pawn),
    square(FX-1, FY-1),
    occupied(FX-1, FY-1, white, _).

% Candidate moves for the moving side
% We need to simulate the board after the move and check if opponent king is in check
% and has no escape.

% Generate candidate moves
candidate_move(Color, pawn, FX, FY, TX, TY) :-
    to_move(Color),
    pawn_push(Color, FX, FY, TX, TY).
candidate_move(Color, pawn, FX, FY, TX, TY) :-
    to_move(Color),
    pawn_capture(Color, FX, FY, TX, TY).

candidate_move(Color, knight, FX, FY, TX, TY) :-
    to_move(Color),
    occupied(FX, FY, Color, knight),
    knight_move(DX, DY),
    TX = FX + DX, TY = FY + DY,
    square(TX, TY),
    not occupied(TX, TY, Color, _).

candidate_move(Color, bishop, FX, FY, TX, TY) :-
    to_move(Color),
    occupied(FX, FY, Color, bishop),
    bishop_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    square(TX, TY),
    not occupied(TX, TY, Color, _),
    FX != TX.

candidate_move(Color, rook, FX, FY, TX, TY) :-
    to_move(Color),
    occupied(FX, FY, Color, rook),
    rook_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    square(TX, TY),
    not occupied(TX, TY, Color, _),
    (FX != TX ; FY != TY).

candidate_move(Color, queen, FX, FY, TX, TY) :-
    to_move(Color),
    occupied(FX, FY, Color, queen),
    queen_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    square(TX, TY),
    not occupied(TX, TY, Color, _),
    FX != TX.
candidate_move(Color, queen, FX, FY, TX, TY) :-
    to_move(Color),
    occupied(FX, FY, Color, queen),
    queen_dir(DX, DY),
    slide_clear(FX, FY, DX, DY, TX, TY),
    square(TX, TY),
    not occupied(TX, TY, Color, _),
    FY != TY.

candidate_move(Color, king, FX, FY, TX, TY) :-
    to_move(Color),
    king(Color, FX, FY),
    king_move(DX, DY),
    TX = FX + DX, TY = FY + DY,
    square(TX, TY),
    not occupied(TX, TY, Color, _).

% After a move (Color, Type, FX, FY, TX, TY), compute new board state
% New position of pieces after move
% piece_after: pieces on board after the move
piece_after(Color, Type, FX, FY, TX, TY, PC, PT, PX, PY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    occupied(PX, PY, PC, PT),
    not (PX = FX, PY = FY, PC = Color, PT = Type),
    not (PX = TX, PY = TY).

% King position after move
king_after(Color, Type, FX, FY, TX, TY, KC, KX, KY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = king, KC = Color, KX = TX, KY = TY.
king_after(Color, Type, FX, FY, TX, TY, KC, KX, KY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(KC, KX, KY),
    not (KC = Color, Type = king).

% After the move, does the opponent's king exist at (KX,KY)?
% We need to check if the opponent king is in check after the move.
% The opponent king position after the move:
opp_king_after(Color, Type, FX, FY, TX, TY, KX, KY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, KX, KY),
    not (TX = KX, TY = KY).  % king not captured (should not happen in legal chess)

% Check if a square is attacked by Color after move (MColor, MType, MFX, MFY, MTX, MTY)
% This is complex - we need to rebuild attack computation on new board
% Simplification: compute attacks based on new piece positions

% New occupied squares after move
new_occ(Color, Type, FX, FY, TX, TY, PX, PY, PC, PT) :-
    piece_after(Color, Type, FX, FY, TX, TY, PC, PT, PX, PY).
% Add the moved piece at new position
new_occ(Color, Type, FX, FY, TX, TY, TX, TY, Color, Type) :-
    candidate_move(Color, Type, FX, FY, TX, TY).

% New king positions
new_king(Color, Type, FX, FY, TX, TY, KC, KX, KY) :-
    king_after(Color, Type, FX, FY, TX, TY, KC, KX, KY).

% Is square (SX,SY) occupied in new board?
new_any_piece(Color, Type, FX, FY, TX, TY, SX, SY) :-
    new_occ(Color, Type, FX, FY, TX, TY, SX, SY, _, _).

% Sliding clear on new board
new_slide_clear(Color, Type, FX, FY, TX, TY, SFX, SFY, DX, DY, STX, STY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    queen_dir(DX, DY),
    STX = SFX + DX, STY = SFY + DY,
    square(SFX, SFY), square(STX, STY).
new_slide_clear(Color, Type, FX, FY, TX, TY, SFX, SFY, DX, DY, STX, STY) :-
    new_slide_clear(Color, Type, FX, FY, TX, TY, SFX, SFY, DX, DY, MX, MY),
    not new_any_piece(Color, Type, FX, FY, TX, TY, MX, MY),
    STX = MX + DX, STY = MY + DY,
    square(STX, STY).

% Attacks by Color on new board after move (MColor, MType, MFX, MFY, MTX, MTY)
new_attacks(MColor, MType, MFX, MFY, MTX, MTY, AColor, rook, AFX, AFY, ATX, ATY) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, AColor, rook),
    rook_dir(DX, DY),
    new_slide_clear(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFX != ATX.
new_attacks(MColor, MType, MFX, MFY, MTX, MTY, AColor, rook, AFX, AFY, ATX, ATY) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, AColor, rook),
    rook_dir(DX, DY),
    new_slide_clear(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFY != ATY.

new_attacks(MColor, MType, MFX, MFY, MTX, MTY, AColor, bishop, AFX, AFY, ATX, ATY) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, AColor, bishop),
    bishop_dir(DX, DY),
    new_slide_clear(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFX != ATX.

new_attacks(MColor, MType, MFX, MFY, MTX, MTY, AColor, queen, AFX, AFY, ATX, ATY) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, AColor, queen),
    queen_dir(DX, DY),
    new_slide_clear(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFX != ATX.
new_attacks(MColor, MType, MFX, MFY, MTX, MTY, AColor, queen, AFX, AFY, ATX, ATY) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, AColor, queen),
    queen_dir(DX, DY),
    new_slide_clear(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFY != ATY.

new_attacks(MColor, MType, MFX, MFY, MTX, MTY, AColor, knight, AFX, AFY, ATX, ATY) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, AColor, knight),
    knight_move(DX, DY),
    ATX = AFX + DX, ATY = AFY + DY,
    square(ATX, ATY).

new_attacks(MColor, MType, MFX, MFY, MTX, MTY, AColor, king, AFX, AFY, ATX, ATY) :-
    new_king(MColor, MType, MFX, MFY, MTX, MTY, AColor, AFX, AFY),
    king_move(DX, DY),
    ATX = AFX + DX, ATY = AFY + DY,
    square(ATX, ATY).

new_attacks(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, AFX, AFY, AFX+1, AFY+1) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, white, pawn),
    square(AFX+1, AFY+1).
new_attacks(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, AFX, AFY, AFX-1, AFY+1) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, white, pawn),
    square(AFX-1, AFY+1).
new_attacks(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, AFX, AFY, AFX+1, AFY-1) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, black, pawn),
    square(AFX+1, AFY-1).
new_attacks(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, AFX, AFY, AFX-1, AFY-1) :-
    new_occ(MColor, MType, MFX, MFY, MTX, MTY, AFX, AFY, black, pawn),
    square(AFX-1, AFY-1).

% After move, is the moving side's own king in check? (illegal move)
own_king_in_check_after(Color, Type, FX, FY, TX, TY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_king(Color, Type, FX, FY, TX, TY, Color, KX, KY),
    new_attacks(Color, Type, FX, FY, TX, TY, Opp, _, _, _, KX, KY).

% Legal move: candidate that doesn't leave own king in check
legal_move(Color, Type, FX, FY, TX, TY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    not own_king_in_check_after(Color, Type, FX, FY, TX, TY).

% After legal move, is opponent king in check?
opp_in_check_after(Color, Type, FX, FY, TX, TY) :-
    legal_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_king(Color, Type, FX, FY, TX, TY, Opp, KX, KY),
    new_attacks(Color, Type, FX, FY, TX, TY, Color, _, _, _, KX, KY).

% Now check if opponent has any legal reply after our move
% Opponent's candidate moves after our move
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    % Opponent pawn pushes
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = pawn,
    Opp = white,
    OTX = OFX, OTY = OFY + 1,
    square(OTX, OTY),
    not new_any_piece(Color, Type, FX, FY, TX, TY, OTX, OTY).
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = pawn,
    Opp = black,
    OTX = OFX, OTY = OFY - 1,
    square(OTX, OTY),
    not new_any_piece(Color, Type, FX, FY, TX, TY, OTX, OTY).

% Pawn double push
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = pawn, Opp = white, OFY = 2,
    OTX = OFX, OTY = 4,
    not new_any_piece(Color, Type, FX, FY, TX, TY, OFX, 3),
    not new_any_piece(Color, Type, FX, FY, TX, TY, OFX, 4).
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = pawn, Opp = black, OFY = 7,
    OTX = OFX, OTY = 5,
    not new_any_piece(Color, Type, FX, FY, TX, TY, OFX, 6),
    not new_any_piece(Color, Type, FX, FY, TX, TY, OFX, 5).

% Pawn captures
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = pawn, Opp = white,
    (OTX = OFX + 1 ; OTX = OFX - 1),
    OTY = OFY + 1,
    square(OTX, OTY),
    new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Color, _).
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = pawn, Opp = black,
    (OTX = OFX + 1 ; OTX = OFX - 1),
    OTY = OFY - 1,
    square(OTX, OTY),
    new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Color, _).

% Knight moves for opponent
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = knight,
    knight_move(DX, DY),
    OTX = OFX + DX, OTY = OFY + DY,
    square(OTX, OTY),
    not new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Opp, _).

% King moves for opponent
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_king(Color, Type, FX, FY, TX, TY, Opp, OFX, OFY),
    OType = king,
    king_move(DX, DY),
    OTX = OFX + DX, OTY = OFY + DY,
    square(OTX, OTY),
    not new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Opp, _).

% Sliding moves for opponent
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = rook,
    rook_dir(DX, DY),
    new_slide_clear(Color, Type, FX, FY, TX, TY, OFX, OFY, DX, DY, OTX, OTY),
    square(OTX, OTY),
    not new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Opp, _),
    OFX != OTX.
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = rook,
    rook_dir(DX, DY),
    new_slide_clear(Color, Type, FX, FY, TX, TY, OFX, OFY, DX, DY, OTX, OTY),
    square(OTX, OTY),
    not new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Opp, _),
    OFY != OTY.

opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = bishop,
    bishop_dir(DX, DY),
    new_slide_clear(Color, Type, FX, FY, TX, TY, OFX, OFY, DX, DY, OTX, OTY),
    square(OTX, OTY),
    not new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Opp, _),
    OFX != OTX.

opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = queen,
    queen_dir(DX, DY),
    new_slide_clear(Color, Type, FX, FY, TX, TY, OFX, OFY, DX, DY, OTX, OTY),
    square(OTX, OTY),
    not new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Opp, _),
    OFX != OTX.
opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    new_occ(Color, Type, FX, FY, TX, TY, OFX, OFY, Opp, OType),
    OType = queen,
    queen_dir(DX, DY),
    new_slide_clear(Color, Type, FX, FY, TX, TY, OFX, OFY, DX, DY, OTX, OTY),
    square(OTX, OTY),
    not new_occ(Color, Type, FX, FY, TX, TY, OTX, OTY, Opp, _),
    OFY != OTY.

% After opponent's reply, compute new board state (2nd level)
% new2_occ: board after opponent's reply
new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, PX, PY, PC, PT) :-
    opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY),
    new_occ(Color, Type, FX, FY, TX, TY, PX, PY, PC, PT),
    not (PX = OFX, PY = OFY),
    not (PX = OTX, PY = OTY).
new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, OTX, OTY, Opp, OType) :-
    opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY),
    opponent(Color, Opp).

new2_king(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, KC, KX, KY) :-
    opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY),
    opponent(Color, Opp),
    OType = king, KC = Opp, KX = OTX, KY = OTY.
new2_king(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, KC, KX, KY) :-
    opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY),
    new_king(Color, Type, FX, FY, TX, TY, KC, KX, KY),
    opponent(Color, Opp),
    not (KC = Opp, OType = king).

new2_any_piece(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, SX, SY) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, SX, SY, _, _).

% Check if opponent's reply leaves their own king in check (illegal)
% After opponent's reply, is opponent's king attacked by moving side?
opp_reply_illegal(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY) :-
    opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY),
    opponent(Color, Opp),
    new2_king(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, Opp, KX, KY),
    % Check if Color attacks KX,KY on new2 board
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, Color, AType),
    new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, Color, AType, AFX, AFY, KX, KY).

% new2 attacks - simplified: just check if Color piece attacks KX,KY
new2_any(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, SX, SY) :-
    new2_any_piece(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, SX, SY).

new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, SFX, SFY, DX, DY, STX, STY) :-
    opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY),
    queen_dir(DX, DY),
    STX = SFX + DX, STY = SFY + DY,
    square(SFX, SFY), square(STX, STY).
new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, SFX, SFY, DX, DY, STX, STY) :-
    new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, SFX, SFY, DX, DY, MX, MY),
    not new2_any(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, MX, MY),
    STX = MX + DX, STY = MY + DY,
    square(STX, STY).

new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, queen, AFX, AFY, ATX, ATY) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, AColor, queen),
    queen_dir(DX, DY),
    new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFX != ATX.
new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, queen, AFX, AFY, ATX, ATY) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, AColor, queen),
    queen_dir(DX, DY),
    new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFY != ATY.

new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, rook, AFX, AFY, ATX, ATY) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, AColor, rook),
    rook_dir(DX, DY),
    new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFX != ATX.
new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, rook, AFX, AFY, ATX, ATY) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, AColor, rook),
    rook_dir(DX, DY),
    new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFY != ATY.

new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, bishop, AFX, AFY, ATX, ATY) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, AColor, bishop),
    bishop_dir(DX, DY),
    new2_slide_clear(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, DX, DY, ATX, ATY),
    square(ATX, ATY),
    AFX != ATX.

new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, knight, AFX, AFY, ATX, ATY) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, AColor, knight),
    knight_move(DX, DY),
    ATX = AFX + DX, ATY = AFY + DY,
    square(ATX, ATY).

new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, king, AFX, AFY, ATX, ATY) :-
    new2_king(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AColor, AFX, AFY),
    king_move(DX, DY),
    ATX = AFX + DX, ATY = AFY + DY,
    square(ATX, ATY).

new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, white, pawn, AFX, AFY, AFX+1, AFY+1) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, white, pawn),
    square(AFX+1, AFY+1).
new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, white, pawn, AFX, AFY, AFX-1, AFY+1) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, white, pawn),
    square(AFX-1, AFY+1).
new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, black, pawn, AFX, AFY, AFX+1, AFY-1) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, black, pawn),
    square(AFX+1, AFY-1).
new2_attacks_color(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, black, pawn, AFX, AFY, AFX-1, AFY-1) :-
    new2_occ(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY, AFX, AFY, black, pawn),
    square(AFX-1, AFY-1).

% A legal opponent reply exists if it's a candidate and not illegal
opp_has_legal_reply(Color, Type, FX, FY, TX, TY) :-
    opp_candidate(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY),
    not opp_reply_illegal(Color, Type, FX, FY, TX, TY, OType, OFX, OFY, OTX, OTY).

% A move is checkmate if:
% 1. It's legal
% 2. It puts opponent in check
% 3. Opponent has no legal reply
is_mate(Color, Type, FX, FY, TX, TY) :-
    legal_move(Color, Type, FX, FY, TX, TY),
    opp_in_check_after(Color, Type, FX, FY, TX, TY),
    not opp_has_legal_reply(Color, Type, FX, FY, TX, TY).

mate_move(Color, Type, FX, FY, TX, TY) :-
    is_mate(Color, Type, FX, FY, TX, TY).

#show mate_move/6.