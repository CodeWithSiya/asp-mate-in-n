%define the chess board
rows(1..8).
cols(1..8).
square(R, C):- rows(R), cols(C).

%column display names
%col_name(1,a). col_name(2,b). col_name(3,c). col_name(4,d).
%col_name(5,e). col_name(6,f). col_name(7,g). col_name(8,h).

%define players
player(white; black).

%define chess pieces
pieceType(king; queen).

%define the initial positions
%placement(player, piece, id, row, col)
placement(white, queen, wq, 7, 2).
placement(white, king, wk, 6, 8).
placement(black, king, bk, 8, 8).

%piece next to move
to_move(white).

%define the legal moves
horizontal(R, C1, R, C2):- square(R, C1), square(R, C2), C1!=C2. %horizontal movement
vertical(R1, C, R2, C):-  square(R1, C), square(R2, C), R1!=R2. %vertical movement
diagonal(R1, C1, R2, C2):- square(R1, C1), square(R2, C2), |R1-R2| = |C1-C2|, (R1, C1) != (R2, C2). %diagonal movement - absolute differences are equal, pairs arent equal

%defining legal moves
%Queen moves - hor, ver, diag - any no. of squares
queen_moves(R1, C1, R2, C2):- horizontal(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- vertical(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- diagonal(R1, C1, R2, C2).

%King moves - to adjacent squares in any direction
king_moves(R1, C1, R2, C2):- square(R1, C1),  square(R2, C2), |C1-C2| <= 1, |R1-R2| <=1, (R1, C1) != (R2, C2).

%---------------------------------------------------------------------------------------------------------------------------
%legal moves - not blocked by another piece, and not occupied by a friendly piece.

%define the pieces that the current piece is between
between_horizontal(R_curr, C_left, C_curr, C_right):- square(R_curr, C_curr), square(R_curr, C_left), square(R_curr, C_right), C_left<C_curr, C_curr<C_right.
between_vertical(C_curr, R_top, R_curr, R_bottom):- square(R_curr, C_curr), square(R_top, C_curr), square(R_bottom, C_curr), R_top<R_curr, R_curr<R_bottom.

between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R1<R_curr, R_curr<R2, C1<C_curr, C_curr<C2. %bottom left to top right diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R2<R_curr, R_curr<R1, C2<C_curr, C_curr<C1. %top right to bottom left diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R1<R_curr, R_curr<R2, C1>C_curr, C_curr>C2. %top left to bottom right diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R2<R_curr, R_curr<R1, C2>C_curr, C_curr>C1. %bottom right to top left diagonal

%movement states
state(before; after).

%defining occupied squares
occupied(before, R, C):- placement(_, _, _, R, C).
occupied(after, R, C):- new_placement(_, R, C).

%define friendly pieces - to stop a player from taking their own pieces
friendly(P, R, C):- placement(P, _, _, R, C).


%defining blocked moves - prevents moving through pieces
% eg - if theres a path from R1, C1 to R2, C2, and there is an occupied square in the way, then the move is blocked
blocked(S, R1, C1, R2, C2):- 
                            between_horizontal(R1, C1, C_curr, C2), 
                            occupied(S, R1, C_curr), R2 = R1.
blocked(S, R1, C1, R2, C2):- 
                            between_horizontal(R1, C2, C_curr, C1), 
                            occupied(S, R1, C_curr), R2 = R1.

blocked(S, R1, C1, R2, C2):- 
                            between_vertical(C1, R1, R_curr, R2), 
                            occupied(S, R_curr, C1), C2 = C1.
blocked(S, R1, C1, R2, C2):- 
                            between_vertical(C1, R2, R_curr, R1), 
                            occupied(S, R_curr, C1), C2 = C1.

blocked(S, R1, C1, R2, C2):- 
                            between_diagonal(R_curr, C_curr, R1, C1, R2, C2), 
                            occupied(S, R_curr, C_curr).


%squares the black king threatens
black_threatens(R2, C2):- 
                        placement(black, king, bk, R1, C1), %where black king is originally
                        king_moves(R1, C1, R2, C2).

%squares the white king threatens
white_threatens(R2, C2):- 
                        placement(white, king, wk, R1, C1),  %where white king is after move
                        king_moves(R1, C1, R2, C2).


%defining legal moves - not blocked. only modelling for white pieces as we want to find a winning move for white, so we assume black will play optimally and move their king out of check if possible.
legal_queen_move(S, R1, C1, R2, C2):- 
                                state(S),
                                queen_moves(R1, C1, R2, C2), %valid move for a queen
                                not blocked(S, R1, C1, R2, C2), % not blocked by another piece
                                not friendly(white, R2, C2). % not occupied by a friendly piece

legal_king_move(P, R1, C1, R2, C2):- 
                                player(P),  
                                king_moves(R1, C1, R2, C2), % valid move for a king
                                not friendly(P, R2, C2).% not occupied by another piece 


%piece must be placed on the board to move
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, queen, ID, R1, C1), 
                                legal_queen_move(before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, king, ID, R1, C1), 
                                legal_king_move(white, R1, C1, R2, C2), 
                                not black_threatens(R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(black, king, ID, R1, C1), 
                                legal_king_move(black, R1, C1, R2, C2), 
                                not white_threatens(R2, C2).

%---------------------------------------------------------------------------------------------------------------------------
% EXTENSION: Mate-in-one reasoning

% 1. Choice rule: select exactly one legal white move
1 { choose_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2), placement(white, _, ID, R1, C1) } 1.

% 2. new_placement: describe the board after the chosen white move
% The moved piece goes to its new square
new_placement(ID, R2, C2) :- choose_move(ID, R1, C1, R2, C2).
% All other white pieces stay where they are
new_placement(ID, R, C) :- placement(white, _, ID, R, C), not choose_move(ID, R, C, _, _).
% Black king stays (unless captured, but king can't be captured in legal chess)
new_placement(bk, R, C) :- placement(black, king, bk, R, C).

% Helper: get piece type and color from ID after move
piece_at_after(white, queen, R, C) :- placement(white, queen, wq, _, _), new_placement(wq, R, C).
piece_at_after(white, king, R, C) :- placement(white, king, wk, _, _), new_placement(wk, R, C).
piece_at_after(black, king, R, C) :- placement(black, king, bk, _, _), new_placement(bk, R, C).

% 3. Threat detection: squares white attacks after the move

% White queen attacks after move
white_queen_attacks_after(R2, C2) :-
    piece_at_after(white, queen, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not blocked(after, R1, C1, R2, C2).

% White king attacks after move
white_king_attacks_after(R2, C2) :-
    piece_at_after(white, king, R1, C1),
    king_moves(R1, C1, R2, C2).

% Combined white attacks after move
white_attacks_after(R, C) :- white_queen_attacks_after(R, C).
white_attacks_after(R, C) :- white_king_attacks_after(R, C).

% 4. Check detection: black king is in check after white's move
black_king_in_check :-
    piece_at_after(black, king, KR, KC),
    white_attacks_after(KR, KC).

% 5. Escape detection

% Black king's position after white's move
black_king_pos(KR, KC) :- piece_at_after(black, king, KR, KC).

% Squares occupied by white pieces after the move (white pieces block black king escape)
white_occupied_after(R, C) :- piece_at_after(white, _, _, R, C).

% A square is friendly to black if it has a black piece (only black king in this position)
black_friendly_after(R, C) :- piece_at_after(black, _, R, C).

% Black king can potentially move to a square if:
% - it's a valid king move
% - not occupied by a black piece
% - within board bounds (already guaranteed by square/2)
potential_king_escape(R2, C2) :-
    black_king_pos(R1, C1),
    king_moves(R1, C1, R2, C2),
    not black_friendly_after(R2, C2).

% After black king moves to (R2,C2), we need to check if white still attacks that square
% We need to recompute white attacks considering the black king has moved away from (R1,C1)
% and possibly captured a white piece at (R2,C2).

% White queen position after white's move
white_queen_pos_after(QR, QC) :- piece_at_after(white, queen, QR, QC).
white_king_pos_after(KR, KC) :- piece_at_after(white, king, KR, KC).

% Check if white queen attacks (R2,C2) after black king moves from (BKR1,BKC1) to (R2,C2)
% The board state: white queen at (QR,QC), white king at (WKR,WKC), black king moved to (R2,C2)
% Black king no longer at (BKR1,BKC1), and if white queen was at (R2,C2) it was captured

% Occupied squares when black king is at escape square (R2,C2):
% - White queen (if not captured): white_queen_pos_after, unless queen was at (R2,C2)
% - White king: white_king_pos_after
% - Black king: at (R2,C2)

% Is a square occupied in the hypothetical position where black king moved to (ER,EC)?
hyp_occupied(ER, EC, R, C) :-
    potential_king_escape(ER, EC),
    % white queen is present if not captured at escape square
    white_queen_pos_after(R, C),
    not (R = ER, C = EC).  % queen not captured

hyp_occupied(ER, EC, R, C) :-
    potential_king_escape(ER, EC),
    white_king_pos_after(R, C).

hyp_occupied(ER, EC, ER, EC) :-
    potential_king_escape(ER, EC).  % black king at escape square

% Is the path from (R1,C1) to (R2,C2) blocked in hypothetical position?
hyp_blocked(ER, EC, R1, C1, R2, C2) :-
    potential_king_escape(ER, EC),
    between_horizontal(R1, C1, C_curr, C2),
    hyp_occupied(ER, EC, R1, C_curr), R2 = R1.
hyp_blocked(ER, EC, R1, C1, R2, C2) :-
    potential_king_escape(ER, EC),
    between_horizontal(R1, C2, C_curr, C1),
    hyp_occupied(ER, EC, R1, C_curr), R2 = R1.
hyp_blocked(ER, EC, R1, C1, R2, C2) :-
    potential_king_escape(ER, EC),
    between_vertical(C1, R1, R_curr, R2),
    hyp_occupied(ER, EC, R_curr, C1), C2 = C1.
hyp_blocked(ER, EC, R1, C1, R2, C2) :-
    potential_king_escape(ER, EC),
    between_vertical(C1, R2, R_curr, R1),
    hyp_occupied(ER, EC, R_curr, C1), C2 = C1.
hyp_blocked(ER, EC, R1, C1, R2, C2) :-
    potential_king_escape(ER, EC),
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    hyp_occupied(ER, EC, R_curr, C_curr).

% White queen attacks escape square (ER,EC) in hypothetical position
% (queen not captured at escape square)
hyp_queen_attacks_escape(ER, EC) :-
    potential_king_escape(ER, EC),
    white_queen_pos_after(QR, QC),
    not (QR = ER, QC = EC),  % queen not captured
    queen_moves(QR, QC, ER, EC),
    not hyp_blocked(ER, EC, QR, QC, ER, EC).

% White king attacks escape square (ER,EC)
hyp_king_attacks_escape(ER, EC) :-
    potential_king_escape(ER, EC),
    white_king_pos_after(WKR, WKC),
    king_moves(WKR, WKC, ER, EC).

% White attacks escape square in hypothetical position
hyp_white_attacks_escape(ER, EC) :-
    hyp_queen_attacks_escape(ER, EC).
hyp_white_attacks_escape(ER, EC) :-
    hyp_king_attacks_escape(ER, EC).

% A king escape is safe if the square is not attacked by white in the hypothetical position
% and not occupied by a white piece (can't move to square occupied by white unless capturing,
% but white queen could be there - that would be a capture, handled by queen not being there)
king_escape_safe(ER, EC) :-
    potential_king_escape(ER, EC),
    not hyp_white_attacks_escape(ER, EC),
    not white_king_pos_after(ER, EC).  % can't capture white king

% Black has an escape if there exists a safe king move
black_has_escape :- king_escape_safe(_, _).

% 6. Checkmate constraint: the chosen move must deliver checkmate
% Reject if black king is not in check
:- not black_king_in_check.

% Reject if black has any escape
:- black_has_escape.

% 7. Identify the chosen move details for output
chosen_move_details(Color, Type, R1, C1, R2, C2) :-
    choose_move(ID, R1, C1, R2, C2),
    placement(Color, Type, ID, R1, C1).

% Output the mating move
mate_move(Color, Type, R1, C1, R2, C2) :-
    chosen_move_details(Color, Type, R1, C1, R2, C2).

#show mate_move/6.