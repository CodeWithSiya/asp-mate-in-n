% Helper: board squares
square(1..8).

% Directions for pieces
rook_dir(1,0). rook_dir(-1,0). rook_dir(0,1). rook_dir(0,-1).
bishop_dir(1,1). bishop_dir(1,-1). bishop_dir(-1,1). bishop_dir(-1,-1).
queen_dir(1,0). queen_dir(-1,0). queen_dir(0,1). queen_dir(0,-1).
queen_dir(1,1). queen_dir(1,-1). queen_dir(-1,1). queen_dir(-1,-1).
king_dir(1,0). king_dir(-1,0). king_dir(0,1). king_dir(0,-1).
king_dir(1,1). king_dir(1,-1). king_dir(-1,1). king_dir(-1,-1).

% Knight moves
knight_move(1,2). knight_move(1,-2). knight_move(-1,2). knight_move(-1,-2).
knight_move(2,1). knight_move(2,-1). knight_move(-2,1). knight_move(-2,-1).

% Opponent color
opponent(white, black). opponent(black, white).

% Sliding piece reachability: squares along a direction up to (but not including) a blocker
% Step k along direction (DX,DY) from (FX,FY)
slide_step(FX, FY, DX, DY, 1, TX, TY) :-
    piece(_, _, FX, FY),
    TX = FX + DX, TY = FY + DY,
    square(TX), square(TY),
    rook_dir(DX, DY).

slide_step(FX, FY, DX, DY, 1, TX, TY) :-
    piece(_, _, FX, FY),
    TX = FX + DX, TY = FY + DY,
    square(TX), square(TY),
    bishop_dir(DX, DY).

slide_step(FX, FY, DX, DY, 1, TX, TY) :-
    piece(_, _, FX, FY),
    TX = FX + DX, TY = FY + DY,
    square(TX), square(TY),
    queen_dir(DX, DY).

% Generate candidate moves for the side to move

% Queen moves (sliding)
can_move(Color, queen, FX, FY, TX, TY) :-
    to_move(Color),
    piece(Color, queen, FX, FY),
    queen_dir(DX, DY),
    slide_to(FX, FY, DX, DY, TX, TY).

% Rook moves (sliding)
can_move(Color, rook, FX, FY, TX, TY) :-
    to_move(Color),
    piece(Color, rook, FX, FY),
    rook_dir(DX, DY),
    slide_to(FX, FY, DX, DY, TX, TY).

% Bishop moves (sliding)
can_move(Color, bishop, FX, FY, TX, TY) :-
    to_move(Color),
    piece(Color, bishop, FX, FY),
    bishop_dir(DX, DY),
    slide_to(FX, FY, DX, DY, TX, TY).

% Knight moves
can_move(Color, knight, FX, FY, TX, TY) :-
    to_move(Color),
    piece(Color, knight, FX, FY),
    knight_move(DX, DY),
    TX = FX + DX, TY = FY + DY,
    square(TX), square(TY),
    not same_color_piece(Color, TX, TY).

% King moves
can_move(Color, king, FX, FY, TX, TY) :-
    to_move(Color),
    king(Color, FX, FY),
    king_dir(DX, DY),
    TX = FX + DX, TY = FY + DY,
    square(TX), square(TY),
    not same_color_piece(Color, TX, TY).

% Pawn moves (white)
can_move(white, pawn, FX, FY, TX, TY) :-
    to_move(white),
    piece(white, pawn, FX, FY),
    TX = FX, TY = FY + 1,
    square(TX), square(TY),
    not any_piece(TX, TY).

can_move(white, pawn, FX, FY, TX, TY) :-
    to_move(white),
    piece(white, pawn, FX, FY),
    FY = 2,
    TX = FX, TY = FY + 2,
    square(TX), square(TY),
    not any_piece(TX, TY),
    not any_piece(FX, FY+1).

can_move(white, pawn, FX, FY, TX, TY) :-
    to_move(white),
    piece(white, pawn, FX, FY),
    TX = FX + 1, TY = FY + 1,
    square(TX), square(TY),
    opponent_piece(white, TX, TY).

can_move(white, pawn, FX, FY, TX, TY) :-
    to_move(white),
    piece(white, pawn, FX, FY),
    TX = FX - 1, TY = FY + 1,
    square(TX), square(TY),
    opponent_piece(white, TX, TY).

% Pawn moves (black)
can_move(black, pawn, FX, FY, TX, TY) :-
    to_move(black),
    piece(black, pawn, FX, FY),
    TX = FX, TY = FY - 1,
    square(TX), square(TY),
    not any_piece(TX, TY).

can_move(black, pawn, FX, FY, TX, TY) :-
    to_move(black),
    piece(black, pawn, FX, FY),
    FY = 7,
    TX = FX, TY = FY - 2,
    square(TX), square(TY),
    not any_piece(TX, TY),
    not any_piece(FX, FY-1).

can_move(black, pawn, FX, FY, TX, TY) :-
    to_move(black),
    piece(black, pawn, FX, FY),
    TX = FX + 1, TY = FY - 1,
    square(TX), square(TY),
    opponent_piece(black, TX, TY).

can_move(black, pawn, FX, FY, TX, TY) :-
    to_move(black),
    piece(black, pawn, FX, FY),
    TX = FX - 1, TY = FY - 1,
    square(TX), square(TY),
    opponent_piece(black, TX, TY).

% Helper: same color piece at square
same_color_piece(Color, X, Y) :- piece(Color, _, X, Y).

% Helper: any piece at square
any_piece(X, Y) :- piece(_, _, X, Y).

% Helper: opponent piece at square
opponent_piece(Color, X, Y) :-
    opponent(Color, Opp),
    piece(Opp, _, X, Y).

% Sliding move generation
% A piece can slide to a square if all intermediate squares are empty and the target is either empty or has an opponent piece

% slide_clear: all squares from step 1 to k-1 along direction are empty
slide_clear(FX, FY, DX, DY, 1) :-
    piece(_, _, FX, FY),
    queen_dir(DX, DY).

slide_clear(FX, FY, DX, DY, 1) :-
    piece(_, _, FX, FY),
    rook_dir(DX, DY).

slide_clear(FX, FY, DX, DY, 1) :-
    piece(_, _, FX, FY),
    bishop_dir(DX, DY).

slide_clear(FX, FY, DX, DY, K) :-
    slide_clear(FX, FY, DX, DY, K1),
    K1 >= 1,
    K = K1 + 1,
    K <= 8,
    IX = FX + DX*(K1), IY = FY + DY*(K1),
    square(IX), square(IY),
    not any_piece(IX, IY).

% slide_to: the piece at (FX,FY) can reach (TX,TY) along direction (DX,DY)
slide_to(FX, FY, DX, DY, TX, TY) :-
    piece(Color, _, FX, FY),
    slide_clear(FX, FY, DX, DY, K),
    TX = FX + DX*K, TY = FY + DY*K,
    square(TX), square(TY),
    not same_color_piece(Color, TX, TY).

% After a move, compute the new board state
% piece_after_move: pieces on board after (Color,Type) moves from (FX,FY) to (TX,TY)
piece_after(Color, Type, FX, FY, TX, TY, Color2, Type2, X2, Y2) :-
    can_move(Color, Type, FX, FY, TX, TY),
    piece(Color2, Type2, X2, Y2),
    not (X2 = FX, Y2 = FY),
    not (X2 = TX, Y2 = TY).

piece_after(Color, Type, FX, FY, TX, TY, Color, Type, TX, TY) :-
    can_move(Color, Type, FX, FY, TX, TY).

% King position after move
king_after(Color, Type, FX, FY, TX, TY, Color, TX, TY) :-
    can_move(Color, Type, FX, FY, TX, TY),
    Type = king.

king_after(Color, Type, FX, FY, TX, TY, Color, KX, KY) :-
    can_move(Color, Type, FX, FY, TX, TY),
    Type != king,
    king(Color, KX, KY).

king_after(Color, Type, FX, FY, TX, TY, Opp, KX, KY) :-
    can_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, KX, KY),
    not (KX = TX, KY = TY).

% Check if a square is attacked by Color after a move
% We need to check if the opponent king is in check after our move

% attacked_after: square (AX,AY) is attacked by Color after move (Color,Type,FX,FY,TX,TY)
% Using the pieces remaining after the move

% For sliding attack checks after move, we need to trace rays from the target square

% Helper: is square (AX,AY) attacked by any piece of Color after the move?
% We'll check each piece type

% Rook/Queen attack along ranks/files after move
rook_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    can_move(Color, Type, FX, FY, TX, TY),
    piece_after(Color, Type, FX, FY, TX, TY, Color, PT, PX, PY),
    (PT = rook ; PT = queen),
    rook_dir(DX, DY),
    slide_attacks_after(Color, Type, FX, FY, TX, TY, PX, PY, DX, DY, AX, AY).

% Bishop/Queen attack along diagonals after move
bishop_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    can_move(Color, Type, FX, FY, TX, TY),
    piece_after(Color, Type, FX, FY, TX, TY, Color, PT, PX, PY),
    (PT = bishop ; PT = queen),
    bishop_dir(DX, DY),
    slide_attacks_after(Color, Type, FX, FY, TX, TY, PX, PY, DX, DY, AX, AY).

% This approach is getting complex. Let me use a simpler direct approach.
% Instead, I'll check if the opponent king is attacked after the move.

% Check if opponent king is in checkmate after our move
% Step 1: Is the opponent king in check after our move?
% Step 2: Can the opponent king escape?

% Let me use a more direct approach for mate detection

% After move (Color, Type, FX, FY, TX, TY), is square (SX,SY) attacked by Color?
% Check queen attacks
sq_attacked_by_queen_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, queen, QX, QY),
    queen_dir(DX, DY),
    ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, SX, SY).

sq_attacked_by_rook_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, rook, QX, QY),
    rook_dir(DX, DY),
    ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, SX, SY).

sq_attacked_by_bishop_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, bishop, QX, QY),
    bishop_dir(DX, DY),
    ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, SX, SY).

sq_attacked_by_knight_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, knight, KNX, KNY),
    knight_move(DX, DY),
    SX = KNX + DX, SY = KNY + DY,
    square(SX), square(SY).

sq_attacked_by_king_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, king, KX, KY),
    king_dir(DX, DY),
    SX = KX + DX, SY = KY + DY,
    square(SX), square(SY).

sq_attacked_by_wpawn_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = white,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, PX, PY),
    (SX = PX + 1 ; SX = PX - 1),
    SY = PY + 1,
    square(SX), square(SY).

sq_attacked_by_bpawn_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = black,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, PX, PY),
    (SX = PX + 1 ; SX = PX - 1),
    SY = PY - 1,
    square(SX), square(SY).

% Combined: square attacked by MColor after move
sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    sq_attacked_by_queen_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY).
sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    sq_attacked_by_rook_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY).
sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    sq_attacked_by_bishop_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY).
sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    sq_attacked_by_knight_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY).
sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    sq_attacked_by_king_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY).
sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    sq_attacked_by_wpawn_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY).
sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    sq_attacked_by_bpawn_after(MColor, MType, MFX, MFY, MTX, MTY, SX, SY).

% ray_reaches_after: a sliding piece at (PX,PY) along (DX,DY) reaches (SX,SY) 
% with no blocking pieces in between (using piece_after board)
ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, SX, SY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(PX), square(PY), square(SX), square(SY),
    SX = PX + DX*K, SY = PY + DY*K,
    K >= 1, K <= 7,
    ray_clear_after(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, K).

% ray_clear_after: steps 1..K-1 along (DX,DY) from (PX,PY) are empty after move
ray_clear_after(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, 1) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(PX), square(PY).

ray_clear_after(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, K) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    K > 1,
    K1 = K - 1,
    ray_clear_after(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, K1),
    IX = PX + DX*K1, IY = PY + DY*K1,
    square(IX), square(IY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, IX, IY).

% piece_after_at: there is some piece at (X,Y) after the move
piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, X, Y) :-
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, _, _, X, Y).

% Opponent king in check after move
king_in_check_after(MColor, MType, MFX, MFY, MTX, MTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    king_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY),
    sq_attacked_after(MColor, MType, MFX, MFY, MTX, MTY, KX, KY).

% Now check if the opponent has any legal escape move after our move
% Opponent's king can move to a square not attacked by us (after our move)
% Also need to check if any opponent piece can block or capture

% After our move (MColor,MType,MFX,MFY,MTX,MTY), opponent tries move (OType,OFX,OFY,OTX,OTY)
% We need to check if after opponent's response, their king is not in check

% Opponent's candidate moves after our move
% This is complex - let's enumerate opponent king moves and other piece moves

% Opponent king escape: king moves to (NKX,NKY) not attacked by MColor after both moves
opp_king_can_escape(MColor, MType, MFX, MFY, MTX, MTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    king_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY),
    king_dir(DX, DY),
    NKX = KX + DX, NKY = KY + DY,
    square(NKX), square(NKY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, NKX, NKY),
    not sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY).

% same_color_after: a piece of Color is at (X,Y) after MColor's move
same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Color, X, Y) :-
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Color, _, X, Y).

% After MColor's move and opponent king moves from (KX,KY) to (NKX,NKY),
% is (NKX,NKY) attacked by MColor?
% Board after both moves: remove king from (KX,KY), place at (NKX,NKY), remove any MColor piece at (NKX,NKY) [already handled]

% We need to check if NKX,NKY is attacked by MColor on the board after:
% 1. MColor's move
% 2. Opponent king moves from KX,KY to NKX,NKY

% For this, we check attacks from MColor pieces (after MColor's move) on (NKX,NKY),
% but the king is no longer at (KX,KY) (it moved), so it can't block

sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    % Check queen attacks
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, queen, QX, QY),
    queen_dir(DX, DY),
    ray_reaches_after2(MColor, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, NKX, NKY, KX, KY, NKX, NKY).

sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, rook, QX, QY),
    rook_dir(DX, DY),
    ray_reaches_after2(MColor, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, NKX, NKY, KX, KY, NKX, NKY).

sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, bishop, QX, QY),
    bishop_dir(DX, DY),
    ray_reaches_after2(MColor, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, NKX, NKY, KX, KY, NKX, NKY).

sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, knight, KNX, KNY),
    knight_move(DDX, DDY),
    NKX = KNX + DDX, NKY = KNY + DDY.

sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, king, MKX, MKY),
    king_dir(DDX, DDY),
    NKX = MKX + DDX, NKY = MKY + DDY.

sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = white,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, PX, PY),
    (NKX = PX + 1 ; NKX = PX - 1),
    NKY = PY + 1.

sq_attacked_after_opp_king_move(MColor, MType, MFX, MFY, MTX, MTY, KX, KY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = black,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, PX, PY),
    (NKX = PX + 1 ; NKX = PX - 1),
    NKY = PY - 1.

% ray_reaches_after2: like ray_reaches_after but with king removed from (KX,KY) and placed at (NKX,NKY)
% The "moved from" square (OKX,OKY) is now empty, "moved to" (NKX,NKY) has the king (but we're checking if it's attacked)
ray_reaches_after2(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, SX, SY, OKX, OKY, NKX, NKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(PX), square(PY), square(SX), square(SY),
    SX = PX + DX*K, SY = PY + DY*K,
    K >= 1, K <= 7,
    ray_clear_after2(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, K, OKX, OKY).

ray_clear_after2(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, 1, OKX, OKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(PX), square(PY).

ray_clear_after2(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, K, OKX, OKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    K > 1,
    K1 = K - 1,
    ray_clear_after2(MColor, MType, MFX, MFY, MTX, MTY, PX, PY, DX, DY, K1, OKX, OKY),
    IX = PX + DX*K1, IY = PY + DY*K1,
    square(IX), square(IY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, IX, IY),
    not (IX = OKX, IY = OKY).
    % Note: king moved away from OKX,OKY so that square is now empty

% Opponent can also block check or capture the checking piece
% For simplicity in mate-in-one, we need to check if opponent has ANY legal move
% that gets their king out of check

% Opponent's other pieces can move to block or capture
% This is very complex to enumerate fully. Let me focus on the key aspects:
% 1. King moves (already handled above)
% 2. Other pieces blocking/capturing

% For a complete solution, let's also check if opponent pieces can capture the checking piece
% or block the check

% Opponent piece (non-king) can move after our move
opp_piece_can_respond(MColor, MType, MFX, MFY, MTX, MTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    % Opponent has a non-king piece
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY),
    OType != king,
    % That piece can move somewhere
    opp_piece_move_valid(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, _, _).

% Valid opponent piece move: results in their king not being in check
opp_piece_move_valid(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    % Opponent queen move
    OType = queen,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, queen, OFX, OFY),
    queen_dir(DDX, DDY),
    opp_slide_to_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OFX, OFY, DDX, DDY, OTX, OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = rook,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, rook, OFX, OFY),
    rook_dir(DDX, DDY),
    opp_slide_to_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OFX, OFY, DDX, DDY, OTX, OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = bishop,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, bishop, OFX, OFY),
    bishop_dir(DDX, DDY),
    opp_slide_to_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OFX, OFY, DDX, DDY, OTX, OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = knight,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, knight, OFX, OFY),
    knight_move(DDX, DDY),
    OTX = OFX + DDX, OTY = OFY + DDY,
    square(OTX), square(OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

% Opponent pawn moves
opp_piece_move_valid(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = pawn,
    Opp = white,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, OFX, OFY),
    OTX = OFX, OTY = OFY + 1,
    square(OTX), square(OTY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, OTX, OTY),
    opp_king_safe_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = pawn,
    Opp = black,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, OFX, OFY),
    OTX = OFX, OTY = OFY - 1,
    square(OTX), square(OTY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, OTX, OTY),
    opp_king_safe_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

% opp_slide_to_after: opponent sliding piece can reach (OTX,OTY) after MColor's move
opp_slide_to_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OFX, OFY, DDX, DDY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opp_slide_clear_after(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, DDX, DDY, K),
    OTX = OFX + DDX*K, OTY = OFY + DDY*K,
    square(OTX), square(OTY).

opp_slide_clear_after(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, DDX, DDY, 1) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(OFX), square(OFY).

opp_slide_clear_after(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, DDX, DDY, K) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    K > 1, K <= 8,
    K1 = K - 1,
    opp_slide_clear_after(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, DDX, DDY, K1),
    IX = OFX + DDX*K1, IY = OFY + DDY*K1,
    square(IX), square(IY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, IX, IY).

% After MColor's move and opponent piece (OType) moves from (OFX,OFY) to (OTX,OTY),
% is the opponent king safe?
opp_king_safe_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    % Find opponent king position after response
    opp_king_pos_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY),
    not sq_attacked_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY).

opp_king_pos_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, OTX, OTY) :-
    OType = king.

opp_king_pos_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, KX, KY) :-
    OType != king,
    king_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY).

% Is (FKX,FKY) attacked by MColor after both moves?
% Board: MColor's move applied, then opponent piece moved from OFX,OFY to OTX,OTY
sq_attacked_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, APT, APX, APY),
    % APT is a queen/rook/bishop
    (APT = queen ; APT = rook),
    rook_dir(DDX, DDY),
    ray_reaches_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY).

sq_attacked_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, APT, APX, APY),
    (APT = queen ; APT = bishop),
    bishop_dir(DDX, DDY),
    ray_reaches_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY).

sq_attacked_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, knight, KNX, KNY),
    knight_move(DDX, DDY),
    FKX = KNX + DDX, FKY = KNY + DDY.

sq_attacked_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, king, MKX, MKY),
    king_dir(DDX, DDY),
    FKX = MKX + DDX, FKY = MKY + DDY.

sq_attacked_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = white,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, PX, PY),
    (FKX = PX + 1 ; FKX = PX - 1),
    FKY = PY + 1.

sq_attacked_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = black,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, PX, PY),
    (FKX = PX + 1 ; FKX = PX - 1),
    FKY = PY - 1.

% ray_reaches_after_resp: ray from (APX,APY) along (DDX,DDY) reaches (FKX,FKY)
% on board after MColor's move, with opponent piece moved from (OFX,OFY) to (OTX,OTY)
ray_reaches_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(APX), square(APY), square(FKX), square(FKY),
    FKX = APX + DDX*K, FKY = APY + DDY*K,
    K >= 1, K <= 7,
    ray_clear_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, K).

ray_clear_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, 1) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(APX), square(APY).

ray_clear_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, K) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    K > 1,
    K1 = K - 1,
    ray_clear_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, K1),
    IX = APX + DDX*K1, IY = APY + DDY*K1,
    square(IX), square(IY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, IX, IY),
    % The opponent piece moved away from OFX,OFY (so that's now empty)
    % and moved to OTX,OTY (so that's now occupied by opponent piece)
    not (IX = OTX, IY = OTY),
    % OFX,OFY is now empty (piece moved away) - already handled by piece_after_at
    % But we need to add: OFX,OFY is empty after opponent's move
    not (IX = OFX, IY = OFY, not piece_after_at_excl(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OFX, OFY)).

% Hmm, this is getting complicated. Let me simplify.
% piece_after_at already reflects MColor's move. After opponent moves OFX,OFY -> OTX,OTY:
% - OFX,OFY becomes empty (unless another piece is there, but there isn't since the piece was there)
% - OTX,OTY gets the opponent piece (possibly capturing MColor piece there)
% For ray clearance, intermediate squares must be empty on this new board.

% Let me redefine ray_clear_after_resp more carefully:
% A square (IX,IY) is occupied on the board after both moves if:
% - It was occupied after MColor's move AND it's not OFX,OFY (opponent piece moved away)
% - OR it's OTX,OTY (opponent piece moved here)

occupied_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, IX, IY) :-
    piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, IX, IY),
    not (IX = OFX, IY = OFY).

occupied_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY).

% Redefine ray_clear_after_resp using occupied_after_resp
ray_clear_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, 1) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(APX), square(APY).

ray_clear_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, K) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    K > 1,
    K1 = K - 1,
    ray_clear_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, K1),
    IX = APX + DDX*K1, IY = APY + DDY*K1,
    square(IX), square(IY),
    not occupied_after_resp(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, IX, IY).

ray_reaches_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    square(APX), square(APY), square(FKX), square(FKY),
    FKX = APX + DDX*K, FKY = APY + DDY*K,
    K >= 1, K <= 7,
    ray_clear_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, K).

% Redefine sq_attacked_after_response using ray_reaches_after_resp2
sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, APT, APX, APY),
    APT = queen,
    rook_dir(DDX, DDY),
    ray_reaches_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY).

sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, APT, APX, APY),
    APT = queen,
    bishop_dir(DDX, DDY),
    ray_reaches_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY).

sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, rook, APX, APY),
    rook_dir(DDX, DDY),
    ray_reaches_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY).

sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, bishop, APX, APY),
    bishop_dir(DDX, DDY),
    ray_reaches_after_resp2(MColor, MType, MFX, MFY, MTX, MTY, OFX, OFY, OTX, OTY, APX, APY, DDX, DDY, FKX, FKY).

sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, knight, KNX, KNY),
    knight_move(DDX, DDY),
    FKX = KNX + DDX, FKY = KNY + DDY.

sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, king, MKX, MKY),
    king_dir(DDX, DDY),
    FKX = MKX + DDX, FKY = MKY + DDY.

sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = white,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, PX, PY),
    (FKX = PX + 1 ; FKX = PX - 1),
    FKY = PY + 1.

sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    MColor = black,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, PX, PY),
    (FKX = PX + 1 ; FKX = PX - 1),
    FKY = PY - 1.

% Redefine opp_king_safe_after_response using sq_attacked_after_response2
opp_king_safe_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opp_king_pos_after_response(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY),
    not sq_attacked_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY, FKX, FKY).

% Redefine opp_piece_move_valid using opp_king_safe_after_response2
opp_piece_can_respond2(MColor, MType, MFX, MFY, MTX, MTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY),
    OType != king,
    opp_piece_move_valid2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY).

opp_piece_move_valid2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = queen,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, queen, OFX, OFY),
    queen_dir(DDX, DDY),
    opp_slide_to_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OFX, OFY, DDX, DDY, OTX, OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = rook,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, rook, OFX, OFY),
    rook_dir(DDX, DDY),
    opp_slide_to_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OFX, OFY, DDX, DDY, OTX, OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = bishop,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, bishop, OFX, OFY),
    bishop_dir(DDX, DDY),
    opp_slide_to_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OFX, OFY, DDX, DDY, OTX, OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = knight,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, knight, OFX, OFY),
    knight_move(DDX, DDY),
    OTX = OFX + DDX, OTY = OFY + DDY,
    square(OTX), square(OTY),
    not same_color_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, OTX, OTY),
    opp_king_safe_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = pawn, Opp = white,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, OFX, OFY),
    OTX = OFX, OTY = OFY + 1,
    square(OTX), square(OTY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, OTX, OTY),
    opp_king_safe_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

opp_piece_move_valid2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    OType = pawn, Opp = black,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, OFX, OFY),
    OTX = OFX, OTY = OFY - 1,
    square(OTX), square(OTY),
    not piece_after_at(MColor, MType, MFX, MFY, MTX, MTY, OTX, OTY),
    opp_king_safe_after_response2(MColor, MType, MFX, MFY, MTX, MTY, Opp, OType, OFX, OFY, OTX, OTY).

% A move is a checkmate if:
% 1. The opponent king is in check after the move
% 2. The opponent king cannot escape (no safe square)
% 3. No opponent piece can block or capture to resolve the check

is_checkmate(MColor, MType, MFX, MFY, MTX, MTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    king_in_check_after(MColor, MType, MFX, MFY, MTX, MTY),
    not opp_king_can_escape(MColor, MType, MFX, MFY, MTX, MTY),
    not opp_piece_can_respond2(MColor, MType, MFX, MFY, MTX, MTY).

% Also need to ensure our own move doesn't leave our king in check
% (i.e., the move is legal)
own_king_in_check_after(MColor, MType, MFX, MFY, MTX, MTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    opponent(MColor, Opp),
    king_after(MColor, MType, MFX, MFY, MTX, MTY, MColor, KX, KY),
    sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY).

% sq_attacked_by_opp_after: is (KX,KY) attacked by Opp after MColor's move?
sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, APT, APX, APY),
    APT = queen,
    rook_dir(DDX, DDY),
    ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, APX, APY, DDX, DDY, KX, KY).

sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, APT, APX, APY),
    APT = queen,
    bishop_dir(DDX, DDY),
    ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, APX, APY, DDX, DDY, KX, KY).

sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, rook, APX, APY),
    rook_dir(DDX, DDY),
    ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, APX, APY, DDX, DDY, KX, KY).

sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, bishop, APX, APY),
    bishop_dir(DDX, DDY),
    ray_reaches_after(MColor, MType, MFX, MFY, MTX, MTY, APX, APY, DDX, DDY, KX, KY).

sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, knight, KNX, KNY),
    knight_move(DDX, DDY),
    KX = KNX + DDX, KY = KNY + DDY.

sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, king, OKX, OKY),
    king_dir(DDX, DDY),
    KX = OKX + DDX, KY = OKY + DDY.

sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    Opp = white,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, white, pawn, PX, PY),
    (KX = PX + 1 ; KX = PX - 1),
    KY = PY + 1.

sq_attacked_by_opp_after(MColor, MType, MFX, MFY, MTX, MTY, Opp, KX, KY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    Opp = black,
    piece_after(MColor, MType, MFX, MFY, MTX, MTY, black, pawn, PX, PY),
    (KX = PX + 1 ; KX = PX - 1),
    KY = PY - 1.

% Legal move: doesn't leave own king in check
legal_move(MColor, MType, MFX, MFY, MTX, MTY) :-
    can_move(MColor, MType, MFX, MFY, MTX, MTY),
    not own_king_in_check_after(MColor, MType, MFX, MFY, MTX, MTY).

% Mate move: legal and results in checkmate
mate_move(Color, Type, FX, FY, TX, TY) :-
    to_move(Color),
    legal_move(Color, Type, FX, FY, TX, TY),
    is_checkmate(Color, Type, FX, FY, TX, TY).

#show mate_move/6.