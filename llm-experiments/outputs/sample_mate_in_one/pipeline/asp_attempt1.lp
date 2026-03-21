% Mate-in-one solver for White

% ============================================================
% Domain definitions
% ============================================================
file(1..8).
rank(1..8).

% Opponent color
opponent(white, black).
opponent(black, white).

% ============================================================
% Occupied squares (any piece including kings)
% ============================================================
occupied(X, Y) :- piece(_, _, X, Y).
occupied(X, Y) :- king(_, X, Y).

occupied_by(Color, X, Y) :- piece(Color, _, X, Y).
occupied_by(Color, X, Y) :- king(Color, X, Y).

% ============================================================
% Blocking helpers for sliding pieces
% ============================================================

% Horizontal blocking: something between (FX,FY) and (TX,FY) on same rank
blocked_h(FX, FY, TX) :-
    occupied(BX, FY),
    FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    occupied(BX, FY),
    FX > TX, BX < FX, BX > TX.

% Vertical blocking: something between (FX,FY) and (FX,TY) on same file
blocked_v(FX, FY, TY) :-
    occupied(FX, BY),
    FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    occupied(FX, BY),
    FY > TY, BY < FY, BY > TY.

% Diagonal blocking: something strictly between (FX,FY) and (TX,TY) on diagonal
% We need TX-FX = TY-FY (same sign diagonal) or TX-FX = FY-TY (opposite sign diagonal)
% Helper: is (BX,BY) strictly between (FX,FY) and (TX,TY) on a diagonal?
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    TX > FX, TY > FY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    TX > FX, TY < FY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    TX < FX, TY > FY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    TX < FX, TY < FY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.

% ============================================================
% Queen move validity (from (FX,FY) to (TX,TY))
% ============================================================
queen_can_move(FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    % same rank
    FY = TY, FX \= TX,
    not blocked_h(FX, FY, TX),
    not occupied_by(Color, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    % same file
    FX = TX, FY \= TY,
    not blocked_v(FX, FY, TY),
    not occupied_by(Color, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    % diagonal: TX-FX = TY-FY (both positive)
    TX > FX, TY > FY,
    TX - FX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    % diagonal: FX-TX = TY-FY (going left-up)
    FX > TX, TY > FY,
    FX - TX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    % diagonal: TX-FX = FY-TY (going right-down)
    TX > FX, FY > TY,
    TX - FX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    % diagonal: FX-TX = FY-TY (going left-down)
    FX > TX, FY > TY,
    FX - TX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

% ============================================================
% Rook move validity
% ============================================================
rook_can_move(FX, FY, TX, TY) :-
    piece(Color, rook, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FY = TY, FX \= TX,
    not blocked_h(FX, FY, TX),
    not occupied_by(Color, TX, TY).

rook_can_move(FX, FY, TX, TY) :-
    piece(Color, rook, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FX = TX, FY \= TY,
    not blocked_v(FX, FY, TY),
    not occupied_by(Color, TX, TY).

% ============================================================
% Bishop move validity
% ============================================================
bishop_can_move(FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX > FX, TY > FY,
    TX - FX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

bishop_can_move(FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FX > TX, TY > FY,
    FX - TX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

bishop_can_move(FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX > FX, FY > TY,
    TX - FX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

bishop_can_move(FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FX > TX, FY > TY,
    FX - TX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(Color, TX, TY).

% ============================================================
% Knight move validity
% ============================================================
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX - FX = 2, TY - FY = 1,
    not occupied_by(Color, TX, TY).
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX - FX = 2, FY - TY = 1,
    not occupied_by(Color, TX, TY).
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FX - TX = 2, TY - FY = 1,
    not occupied_by(Color, TX, TY).
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FX - TX = 2, FY - TY = 1,
    not occupied_by(Color, TX, TY).
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX - FX = 1, TY - FY = 2,
    not occupied_by(Color, TX, TY).
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX - FX = 1, FY - TY = 2,
    not occupied_by(Color, TX, TY).
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FX - TX = 1, TY - FY = 2,
    not occupied_by(Color, TX, TY).
knight_can_move(FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    FX - TX = 1, FY - TY = 2,
    not occupied_by(Color, TX, TY).

% ============================================================
% Pawn move validity (White moves up: rank increases)
% ============================================================
pawn_can_move(FX, FY, TX, TY) :-
    piece(white, pawn, FX, FY),
    to_move(white),
    TX = FX, TY = FY + 1,
    file(TX), rank(TY),
    not occupied(TX, TY).

pawn_can_move(FX, FY, TX, TY) :-
    piece(white, pawn, FX, FY),
    to_move(white),
    FY = 2, TX = FX, TY = 4,
    file(TX), rank(TY),
    not occupied(FX, 3),
    not occupied(FX, 4).

% Pawn captures (White)
pawn_can_move(FX, FY, TX, TY) :-
    piece(white, pawn, FX, FY),
    to_move(white),
    TY = FY + 1,
    TX = FX + 1,
    file(TX), rank(TY),
    occupied_by(black, TX, TY).

pawn_can_move(FX, FY, TX, TY) :-
    piece(white, pawn, FX, FY),
    to_move(white),
    TY = FY + 1,
    FX = TX + 1,
    file(TX), rank(TY),
    occupied_by(black, TX, TY).

% Black pawn moves (moves down: rank decreases)
pawn_can_move(FX, FY, TX, TY) :-
    piece(black, pawn, FX, FY),
    to_move(black),
    TX = FX, TY = FY - 1,
    file(TX), rank(TY),
    not occupied(TX, TY).

pawn_can_move(FX, FY, TX, TY) :-
    piece(black, pawn, FX, FY),
    to_move(black),
    FY = 7, TX = FX, TY = 5,
    file(TX), rank(TY),
    not occupied(FX, 6),
    not occupied(FX, 5).

pawn_can_move(FX, FY, TX, TY) :-
    piece(black, pawn, FX, FY),
    to_move(black),
    TY = FY - 1,
    TX = FX + 1,
    file(TX), rank(TY),
    occupied_by(white, TX, TY).

pawn_can_move(FX, FY, TX, TY) :-
    piece(black, pawn, FX, FY),
    to_move(black),
    TY = FY - 1,
    FX = TX + 1,
    file(TX), rank(TY),
    occupied_by(white, TX, TY).

% ============================================================
% King move validity (moving side)
% ============================================================
king_can_move(FX, FY, TX, TY) :-
    king(Color, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX - FX <= 1, FX - TX <= 1,
    TY - FY <= 1, FY - TY <= 1,
    TX \= FX,
    not occupied_by(Color, TX, TY).

king_can_move(FX, FY, TX, TY) :-
    king(Color, FX, FY),
    to_move(Color),
    file(TX), rank(TY),
    TX - FX <= 1, FX - TX <= 1,
    TY - FY <= 1, FY - TY <= 1,
    TY \= FY,
    not occupied_by(Color, TX, TY).

% ============================================================
% All candidate moves for the moving side
% ============================================================
candidate_move(Color, queen, FX, FY, TX, TY) :-
    to_move(Color),
    queen_can_move(FX, FY, TX, TY).

candidate_move(Color, rook, FX, FY, TX, TY) :-
    to_move(Color),
    rook_can_move(FX, FY, TX, TY).

candidate_move(Color, bishop, FX, FY, TX, TY) :-
    to_move(Color),
    bishop_can_move(FX, FY, TX, TY).

candidate_move(Color, knight, FX, FY, TX, TY) :-
    to_move(Color),
    knight_can_move(FX, FY, TX, TY).

candidate_move(Color, pawn, FX, FY, TX, TY) :-
    to_move(Color),
    pawn_can_move(FX, FY, TX, TY).

candidate_move(Color, king, FX, FY, TX, TY) :-
    to_move(Color),
    king_can_move(FX, FY, TX, TY).

% ============================================================
% After a candidate move, compute the resulting board state
% ============================================================

% Pieces after move: remove captured piece at (TX,TY), move piece from (FX,FY) to (TX,TY)
piece_after(Color, Type, FX, FY, TX, TY, PC, PT, PX, PY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(PC, PT, PX, PY),
    % not the piece being moved
    not (PC = Color, PT = Type, PX = FX, PY = FY),
    % not a piece captured at destination
    not (PX = TX, PY = TY).

% King position after move
king_after(Color, Type, FX, FY, TX, TY, KC, KX, KY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(KC, KX, KY),
    % if moving piece is the king of Color, update its position
    not (Type = king, KC = Color).

% Moving king's new position
king_after_pos(Color, TX, TY) :-
    candidate_move(Color, king, FX, FY, TX, TY).

% Non-moving king stays
king_after_pos_other(Color, Type, FX, FY, TX, TY, KC, KX, KY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(KC, KX, KY),
    KC \= Color.

% ============================================================
% Attack detection after a move
% We need to check if the opponent's king is in check after our move,
% and if the opponent has no legal responses.
%
% Strategy: For each candidate move (Color, Type, FX,FY,TX,TY),
% simulate the board and check if opponent king is in checkmate.
% ============================================================

% After move (Color,Type,FX,FY,TX,TY), is square (SX,SY) occupied?
sq_occupied_after(Color, Type, FX, FY, TX, TY, SX, SY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece_after(Color, Type, FX, FY, TX, TY, _, _, SX, SY).

sq_occupied_after(Color, Type, FX, FY, TX, TY, SX, SY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    % moving piece lands on TX,TY
    SX = TX, SY = TY.

sq_occupied_after(Color, Type, FX, FY, TX, TY, SX, SY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(KC, KX, KY),
    KC \= Color,
    SX = KX, SY = KY.

sq_occupied_after(Color, Type, FX, FY, TX, TY, SX, SY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(Color, KX, KY),
    Type \= king,
    SX = KX, SY = KY.

sq_occupied_after(Color, Type, FX, FY, TX, TY, SX, SY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = king,
    SX = TX, SY = TY.

% ============================================================
% After the move, does Color's piece at (TX,TY) [or remaining pieces]
% attack the opponent king?
% We need: after move, is opponent king in check?
% ============================================================

% Opponent king position after move (it doesn't move, unless captured)
opp_king_pos_after(Color, Type, FX, FY, TX, TY, OKX, OKY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, OKX, OKY),
    % king not captured (if it were, it's checkmate trivially)
    not (TX = OKX, TY = OKY).

% ============================================================
% Check if a square is attacked by Color after move (FX,FY)->(TX,TY)
% We check attacks by the moving side (Color) on the opponent king square
% ============================================================

% Helper: is (BX,BY) occupied after move, excluding the king being checked?
% We use sq_occupied_after but need to be careful.

% Blocking on rank after move
blocked_h_after(Color, Type, FX, FY, TX, TY, SF, SR, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, BX, SR),
    SF < TR, BX > SF, BX < TR.
blocked_h_after(Color, Type, FX, FY, TX, TY, SF, SR, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, BX, SR),
    SF > TR, BX < SF, BX > TR.

blocked_v_after(Color, Type, FX, FY, TX, TY, SF, SR, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, SF, BY),
    SR < TR, BY > SR, BY < TR.
blocked_v_after(Color, Type, FX, FY, TX, TY, SF, SR, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, SF, BY),
    SR > TR, BY < SR, BY > TR.

blocked_d_after(Color, Type, FX, FY, TX, TY, SF, SR, TF, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, BX, BY),
    TF > SF, TR > SR,
    BX > SF, BX < TF, BY > SR, BY < TR,
    BX - SF = BY - SR.
blocked_d_after(Color, Type, FX, FY, TX, TY, SF, SR, TF, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, BX, BY),
    SF > TF, TR > SR,
    BX < SF, BX > TF, BY > SR, BY < TR,
    SF - BX = BY - SR.
blocked_d_after(Color, Type, FX, FY, TX, TY, SF, SR, TF, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, BX, BY),
    TF > SF, SR > TR,
    BX > SF, BX < TF, BY < SR, BY > TR,
    BX - SF = SR - BY.
blocked_d_after(Color, Type, FX, FY, TX, TY, SF, SR, TF, TR) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    sq_occupied_after(Color, Type, FX, FY, TX, TY, BX, BY),
    SF > TF, SR > TR,
    BX < SF, BX > TF, BY < SR, BY > TR,
    SF - BX = SR - BY.

% ============================================================
% Does Color's queen (now at TX,TY after move) attack square (AX,AY)?
% ============================================================
queen_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    % same rank
    TY = AY, TX \= AX,
    not blocked_h_after(Color, Type, FX, FY, TX, TY, TX, TY, AX).

queen_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    % same file
    TX = AX, TY \= AY,
    not blocked_v_after(Color, Type, FX, FY, TX, TY, TX, TY, AY).

queen_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    AX > TX, AY > TY,
    AX - TX = AY - TY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, TX, TY, AX, AY).

queen_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    TX > AX, AY > TY,
    TX - AX = AY - TY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, TX, TY, AX, AY).

queen_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    AX > TX, TY > AY,
    AX - TX = TY - AY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, TX, TY, AX, AY).

queen_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    TX > AX, TY > AY,
    TX - AX = TY - AY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, TX, TY, AX, AY).

% ============================================================
% Does any remaining Color piece attack (AX,AY) after move?
% ============================================================

% Remaining queen (not the one that moved, if any)
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    % same rank
    QY = AY, QX \= AX,
    not blocked_h_after(Color, Type, FX, FY, TX, TY, QX, QY, AX).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    % same file
    QX = AX, QY \= AY,
    not blocked_v_after(Color, Type, FX, FY, TX, TY, QX, QY, AY).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    AX > QX, AY > QY, AX - QX = AY - QY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, QX, QY, AX, AY).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    QX > AX, AY > QY, QX - AX = AY - QY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, QX, QY, AX, AY).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    AX > QX, QY > AY, AX - QX = QY - AY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, QX, QY, AX, AY).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    QX > AX, QY > AY, QX - AX = QY - AY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, QX, QY, AX, AY).

% Remaining rook attacks
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, rook, RX, RY),
    not (RX = FX, RY = FY, Type = rook),
    not (RX = TX, RY = TY),
    RY = AY, RX \= AX,
    not blocked_h_after(Color, Type, FX, FY, TX, TY, RX, RY, AX).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, rook, RX, RY),
    not (RX = FX, RY = FY, Type = rook),
    not (RX = TX, RY = TY),
    RX = AX, RY \= AY,
    not blocked_v_after(Color, Type, FX, FY, TX, TY, RX, RY, AY).

% Remaining bishop attacks
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    AX > BPX, AY > BPY, AX - BPX = AY - BPY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, BPX, BPY, AX, AY).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    BPX > AX, AY > BPY, BPX - AX = AY - BPY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, BPX, BPY, AX, AY).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    AX > BPX, BPY > AY, AX - BPX = BPY - AY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, BPX, BPY, AX, AY).

piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    BPX > AX, BPY > AY, BPX - AX = BPY - AY,
    not blocked_d_after(Color, Type, FX, FY, TX, TY, BPX, BPY, AX, AY).

% Remaining knight attacks
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    AX - NX = 2, AY - NY = 1.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    AX - NX = 2, NY - AY = 1.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - AX = 2, AY - NY = 1.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - AX = 2, NY - AY = 1.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    AX - NX = 1, AY - NY = 2.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    AX - NX = 1, NY - AY = 2.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - AX = 1, AY - NY = 2.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - AX = 1, NY - AY = 2.

% White pawn attacks (attacks diagonally forward = rank+1)
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = white,
    piece(white, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    AY = PY + 1,
    AX = PX + 1.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = white,
    piece(white, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    AY = PY + 1,
    PX = AX + 1.

% Black pawn attacks (attacks diagonally forward = rank-1)
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = black,
    piece(black, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    PY = AY + 1,
    AX = PX + 1.
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = black,
    piece(black, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    PY = AY + 1,
    PX = AX + 1.

% Moving piece attacks (already at TX,TY)
piece_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    queen_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY).

% White king attacks (for completeness - the moving king)
king_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = king,
    AX - TX <= 1, TX - AX <= 1,
    AY - TY <= 1, TY - AY <= 1,
    AX \= TX.
king_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = king,
    AX - TX <= 1, TX - AX <= 1,
    AY - TY <= 1, TY - AY <= 1,
    AY \= TY.

% Non-moving king attacks
king_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(Color, KX, KY),
    Type \= king,
    AX - KX <= 1, KX - AX <= 1,
    AY - KY <= 1, KY - AY <= 1,
    AX \= KX.
king_attacks_after(Color, Type, FX, FY, TX, TY, AX, AY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(Color, KX, KY),
    Type \= king,
    AX - KX <= 1, KX - AX <= 1,
    AY - KY <= 1, KY - AY <= 1,
    AY \= KY.

% ============================================================
% Is the opponent king in check after move (Color,Type,FX,FY,TX,TY)?
% ============================================================
opp_king_in_check_after(Color, Type, FX, FY, TX, TY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, OKX, OKY),
    % Check if opponent king is captured
    TX = OKX, TY = OKY.

opp_king_in_check_after(Color, Type, FX, FY, TX, TY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, OKX, OKY),
    TX \= OKX,
    piece_attacks_after(Color, Type, FX, FY, TX, TY, OKX, OKY).

opp_king_in_check_after(Color, Type, FX, FY, TX, TY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, OKX, OKY),
    TY \= OKY,
    piece_attacks_after(Color, Type, FX, FY, TX, TY, OKX, OKY).

opp_king_in_check_after(Color, Type, FX, FY, TX, TY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, OKX, OKY),
    king_attacks_after(Color, Type, FX, FY, TX, TY, OKX, OKY).

% ============================================================
% Now check if the opponent can escape: simulate opponent's responses
% ============================================================

% After move (Color,Type,FX,FY,TX,TY), the board state is defined.
% Opponent tries to move their king to (EX,EY).
% We need to check if (EX,EY) is safe (not attacked by Color).

% Is square (SX,SY) attacked by Color after move (Color,Type,FX,FY,TX,TY),
% considering the opponent king has moved away from (OKX,OKY) to (EX,EY)?
% This is complex - we need to check if the escape square is attacked.

% Simplified approach: check if opponent king can move to any adjacent square
% that is not occupied by opponent pieces and not attacked by Color.

% Opponent king escape: king moves from (OKX,OKY) to (EX,EY)
opp_king_escape(Color, Type, FX, FY, TX, TY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, OKX, OKY),
    file(EX), rank(EY),
    EX - OKX <= 1, OKX - EX <= 1,
    EY - OKY <= 1, OKY - EY <= 1,
    % not same square
    EX \= OKX,
    % not occupied by opponent piece (own pieces block)
    not occupied_by(Opp, EX, EY),
    % not the square where the moving piece just landed (unless capturing it)
    % actually if EX=TX,EY=TY that means king captures the piece - allowed if not defended
    % not attacked by Color after the move (with king having moved)
    not sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY).

opp_king_escape(Color, Type, FX, FY, TX, TY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opponent(Color, Opp),
    king(Opp, OKX, OKY),
    file(EX), rank(EY),
    EX - OKX <= 1, OKX - EX <= 1,
    EY - OKY <= 1, OKY - EY <= 1,
    % not same square
    EY \= OKY,
    not occupied_by(Opp, EX, EY),
    not sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY).

% Is (EX,EY) attacked by Color, given that opponent king moved from (OKX,OKY)?
% The board after Color's move, with opponent king now at (EX,EY) instead of (OKX,OKY).

% For sliding pieces, the opponent king's old position is now empty.
% We need to recheck blocking without the king at (OKX,OKY).

% Occupied squares after Color's move AND after opponent king moves to (EX,EY)
sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SX, SY) :-
    sq_occupied_after(Color, Type, FX, FY, TX, TY, SX, SY),
    % remove old king position
    not (SX = OKX, SY = OKY),
    % remove new king position from "other" pieces (king is now at EX,EY but we track separately)
    not (SX = EX, SY = EY).

% Blocking after king move
blocked_h_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BX, SR),
    SF < TR, BX > SF, BX < TR.
blocked_h_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BX, SR),
    SF > TR, BX < SF, BX > TR.

blocked_v_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, BY),
    SR < TR, BY > SR, BY < TR.
blocked_v_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, BY),
    SR > TR, BY < SR, BY > TR.

blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TF, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BX, BY),
    TF > SF, TR > SR, BX > SF, BX < TF, BY > SR, BY < TR, BX - SF = BY - SR.
blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TF, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BX, BY),
    SF > TF, TR > SR, BX < SF, BX > TF, BY > SR, BY < TR, SF - BX = BY - SR.
blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TF, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BX, BY),
    TF > SF, SR > TR, BX > SF, BX < TF, BY < SR, BY > TR, BX - SF = SR - BY.
blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, SF, SR, TF, TR) :-
    sq_occ_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BX, BY),
    SF > TF, SR > TR, BX < SF, BX > TF, BY < SR, BY > TR, SF - BX = SR - BY.

% Is (EX,EY) attacked by Color's queen at (TX,TY) after king moves?
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    % same rank
    TY = EY, TX \= EX,
    not blocked_h_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, TX, TY, EX).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    TX = EX, TY \= EY,
    not blocked_v_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, TX, TY, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    EX > TX, EY > TY, EX - TX = EY - TY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, TX, TY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    TX > EX, EY > TY, TX - EX = EY - TY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, TX, TY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    EX > TX, TY > EY, EX - TX = TY - EY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, TX, TY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = queen,
    TX > EX, TY > EY, TX - EX = TY - EY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, TX, TY, EX, EY).

% Remaining queen attacks after king move
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    QY = EY, QX \= EX,
    not blocked_h_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, QX, QY, EX).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    QX = EX, QY \= EY,
    not blocked_v_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, QX, QY, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    EX > QX, EY > QY, EX - QX = EY - QY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, QX, QY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    QX > EX, EY > QY, QX - EX = EY - QY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, QX, QY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    EX > QX, QY > EY, EX - QX = QY - EY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, QX, QY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, queen, QX, QY),
    not (QX = FX, QY = FY, Type = queen),
    not (QX = TX, QY = TY),
    QX > EX, QY > EY, QX - EX = QY - EY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, QX, QY, EX, EY).

% Rook attacks after king move
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, rook, RX, RY),
    not (RX = FX, RY = FY, Type = rook),
    not (RX = TX, RY = TY),
    RY = EY, RX \= EX,
    not blocked_h_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, RX, RY, EX).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, rook, RX, RY),
    not (RX = FX, RY = FY, Type = rook),
    not (RX = TX, RY = TY),
    RX = EX, RY \= EY,
    not blocked_v_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, RX, RY, EY).

% Bishop attacks after king move
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    EX > BPX, EY > BPY, EX - BPX = EY - BPY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BPX, BPY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    BPX > EX, EY > BPY, BPX - EX = EY - BPY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BPX, BPY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    EX > BPX, BPY > EY, EX - BPX = BPY - EY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BPX, BPY, EX, EY).

sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, bishop, BPX, BPY),
    not (BPX = FX, BPY = FY, Type = bishop),
    not (BPX = TX, BPY = TY),
    BPX > EX, BPY > EY, BPX - EX = BPY - EY,
    not blocked_d_after2(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY, BPX, BPY, EX, EY).

% Knight attacks after king move
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    EX - NX = 2, EY - NY = 1.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    EX - NX = 2, NY - EY = 1.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - EX = 2, EY - NY = 1.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - EX = 2, NY - EY = 1.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    EX - NX = 1, EY - NY = 2.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    EX - NX = 1, NY - EY = 2.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - EX = 1, EY - NY = 2.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    piece(Color, knight, NX, NY),
    not (NX = FX, NY = FY, Type = knight),
    not (NX = TX, NY = TY),
    NX - EX = 1, NY - EY = 2.

% Pawn attacks after king move (white pawns)
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = white,
    piece(white, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    EY = PY + 1, EX = PX + 1.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = white,
    piece(white, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    EY = PY + 1, PX = EX + 1.

% Pawn attacks after king move (black pawns)
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = black,
    piece(black, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    PY = EY + 1, EX = PX + 1.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Color = black,
    piece(black, pawn, PX, PY),
    not (PX = FX, PY = FY, Type = pawn),
    not (PX = TX, PY = TY),
    PY = EY + 1, PX = EX + 1.

% Moving king attacks after king move
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = king,
    EX - TX <= 1, TX - EX <= 1,
    EY - TY <= 1, TY - EY <= 1,
    EX \= TX.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    Type = king,
    EX - TX <= 1, TX - EX <= 1,
    EY - TY <= 1, TY - EY <= 1,
    EY \= TY.

% Non-moving king attacks after king move
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(Color, KX, KY),
    Type \= king,
    EX - KX <= 1, KX - EX <= 1,
    EY - KY <= 1, KY - EY <= 1,
    EX \= KX.
sq_attacked_by_after_king_move(Color, Type, FX, FY, TX, TY, OKX, OKY, EX, EY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    king(Color, KX, KY),
    Type \= king,
    EX - KX <= 1, KX - EX <= 1,
    EY - KY <= 1, KY - EY <= 1,
    EY \= KY.

% ============================================================
% Checkmate: opponent king is in check and has no escape
% ============================================================
is_checkmate(Color, Type, FX, FY, TX, TY) :-
    candidate_move(Color, Type, FX, FY, TX, TY),
    opp_king_in_check_after(Color, Type, FX, FY, TX, TY),
    not opp_king_escape(Color, Type, FX, FY, TX, TY, _, _).

% ============================================================
% Output
% ============================================================
mate_move(Color, Type, FX, FY, TX, TY) :-
    is_checkmate(Color, Type, FX, FY, TX, TY).

#show mate_move/6.